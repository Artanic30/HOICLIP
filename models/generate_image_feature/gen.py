import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from util.misc import nested_tensor_from_tensor_list


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)


class GEN(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, num_queries=64, clip_dim=768):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        instance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation, normalize_before)
        instance_decoder_norm = LayerNorm(d_model)
        self.instance_decoder = TransformerDecoder(instance_decoder_layer,
                                                   num_dec_layers,
                                                   instance_decoder_norm,
                                                   return_intermediate=return_intermediate_dec)

        interaction_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before)
        interaction_decoder_norm = LayerNorm(d_model)
        self.interaction_decoder = TransformerDecoder(interaction_decoder_layer,
                                                      num_dec_layers,
                                                      interaction_decoder_norm,
                                                      return_intermediate=return_intermediate_dec)

        clip_interaction_decoder_layer = TransformerDecoderLayer(clip_dim, nhead, dim_feedforward,
                                                                 dropout, activation, normalize_before)
        clip_interaction_decoder_norm = LayerNorm(clip_dim)
        self.clip_interaction_decoder = TransformerDecoderCLIP(clip_interaction_decoder_layer,
                                                               num_dec_layers,
                                                               clip_interaction_decoder_norm,
                                                               return_intermediate=return_intermediate_dec)
        self.inter_guided_embedd = nn.Embedding(num_queries, clip_dim)
        self.queries2spacial_proj = nn.Linear(d_model, clip_dim)
        self.queries2spacial_proj_norm = LayerNorm(clip_dim)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.inter_guided_embedd.weight)

    def forward(self, src, mask, query_embed_h, query_embed_o, pos_guided_embed, pos_embed, clip_model, clip_proj,
                clip_src):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        num_queries = query_embed_h.shape[0]

        query_embed_o = query_embed_o + pos_guided_embed
        query_embed_h = query_embed_h + pos_guided_embed
        query_embed_o = query_embed_o.unsqueeze(1).repeat(1, bs, 1)
        query_embed_h = query_embed_h.unsqueeze(1).repeat(1, bs, 1)
        ins_query_embed = torch.cat((query_embed_h, query_embed_o), dim=0)

        mask = mask.flatten(1)
        ins_tgt = torch.zeros_like(ins_query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        ins_hs = self.instance_decoder(ins_tgt, memory, memory_key_padding_mask=mask,
                                       pos=pos_embed, query_pos=ins_query_embed)
        ins_hs = ins_hs.transpose(1, 2)
        h_hs = ins_hs[:, :, :num_queries, :]
        o_hs = ins_hs[:, :, num_queries:, :]

        inter_hs = (h_hs + o_hs) / 2.0
        inter_hs = self.queries2spacial_proj(inter_hs[-1])
        inter_hs = self.queries2spacial_proj_norm(inter_hs)
        # inter_hs = inter_hs + self.inter_guided_embedd.weight.unsqueeze(0).repeat(bs, 1, 1)

        clip_cls_feature, clip_visual = clip_model.encode_image(clip_src)
        clip_pos = clip_model.visual.positional_embedding.unsqueeze(0).repeat(bs, 1, 1)

        clip_cls_feature = clip_cls_feature / clip_cls_feature.norm(dim=1, keepdim=True)
        clip_cls_feature = clip_cls_feature.unsqueeze(1).repeat(1, num_queries, 1)

        inter_tgt = torch.zeros_like(inter_hs[0])
        dtype = inter_tgt.dtype
        inter_hs = self.clip_interaction_decoder(inter_hs.permute(1, 0, 2).to(dtype),
                                                 clip_visual.permute(1, 0, 2).to(dtype), pos=clip_pos.permute(1, 0, 2))
        inter_hs = inter_hs @ clip_proj.to(dtype)
        inter_hs = inter_hs.permute(0, 2, 1, 3)

        # add
        # ins_guided_embed = (h_hs + o_hs) / 2.0
        # ins_guided_embed = ins_guided_embed.permute(0, 2, 1, 3)
        # #torch.Size([3, 64, 8, 256])
        #
        # inter_tgt = torch.zeros_like(ins_guided_embed[0])
        # inter_hs = self.interaction_decoder(inter_tgt, memory, memory_key_padding_mask=mask,
        #                                     pos=pos_embed, query_pos=ins_guided_embed)
        # inter_hs = inter_hs.transpose(1, 2)

        return h_hs, o_hs, inter_hs, clip_cls_feature


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderCLIP(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(output.shape) == 4:
                output = output[i]
            else:
                # only this branch will be used, we only use last human/object query and pass one layer decoder block
                output = output
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            if len(query_pos.shape) == 4:
                this_query_pos = query_pos[i]
            else:
                this_query_pos = query_pos
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=this_query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_gen(args):
    return GEN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
