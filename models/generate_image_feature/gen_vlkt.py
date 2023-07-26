import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
from datasets.static_hico import HOI_IDX_TO_ACT_IDX

from ..backbone import build_backbone
from ..matcher import build_matcher
from .gen import build_gen
from PIL import Image
import torchvision.transforms as T


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class GEN_VLKT(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()
        self.args = args
        self.clip_model, self.preprocess = clip.load(self.args.clip_model)

        self.obj_feature = nn.Parameter(torch.zeros(81, 512))
        self.hoi_feature = nn.Parameter(torch.zeros(600, 512))
        self.verb_feature = nn.Parameter(torch.zeros(117, 512))


    def forward(self, samples: NestedTensor, is_training=True, clip_input=None, targets=None):
        for t in targets:
            if t['obj_boxes'].shape[0] == 0 or t['human_img'].shape[0] == 0 or t['hoi_area_img'].shape[0] == 0:
                continue
            # print(f"human_img: {t['human_img'].shape}")
            h_feature = self.clip_model.encode_image(t['human_img'])[0]
            o_feature = self.clip_model.encode_image(t['object_img'])[0]
            hoi_feature = self.clip_model.encode_image(t['hoi_area_img'])[0]
            verb_feature = hoi_feature.clone() * 2 - o_feature.clone() - h_feature.clone()

            # h_feature = h_feature / h_feature.norm(dim=1, keepdim=True)
            # o_feature = o_feature / o_feature.norm(dim=1, keepdim=True)
            # hoi_feature = hoi_feature / hoi_feature.norm(dim=1, keepdim=True)
            if h_feature.shape[0] != o_feature.shape[0] or h_feature.shape[0] != hoi_feature.shape[0]:
                raise ValueError

            obj_label = t['obj_cls']
            hoi_label = t['hoi_cls'] - 1
            if obj_label.shape[0] != o_feature.shape[0] or hoi_label.shape[0] != hoi_feature.shape[0]:
                raise ValueError

            if obj_label.max() > 80 or hoi_label.max() >= 600:
                raise ValueError
            ver_label = torch.tensor(HOI_IDX_TO_ACT_IDX)[hoi_label]

            # print(f"obj_label: {obj_label}")
            # print(f"hoi_label: {hoi_label}")



            if torch.isnan(o_feature.sum()) or torch.isnan(hoi_feature.sum()) or torch.isnan(h_feature.sum()):
                print(obj_label)
                print(hoi_label)
                continue

            if 66 in hoi_label or 166 in hoi_label:
                print(hoi_label)
                print(hoi_feature)

            self.obj_feature.data[0] += h_feature.mean(dim=0)
            self.obj_feature.data[obj_label] += o_feature
            self.hoi_feature.data[hoi_label] += hoi_feature
            self.verb_feature.data[ver_label] += verb_feature

        return self.obj_feature.data, self.hoi_feature.data, self.verb_feature.data


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = GEN_VLKT(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    if args.with_clip_label:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.with_rec_loss:
        weight_dict['loss_rec'] = args.rec_loss_coef


    return model, model, model
