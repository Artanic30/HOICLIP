import argparse
import datetime
import json
import random
import time
from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
import os

from util.scheduler import CosineAnnealingLRWarmup, MultiStepLRWarmup


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_clip', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval_each', default=4, type=int)
    parser.add_argument('--eval_each_lr_drop', default=2, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of stage1 decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--with_mimic', action='store_true',
                        help="Use clip feature mimic")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_hoi', default=1, type=float,
                        help="Hoi class coefficient")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--hoi_loss_coef', default=2, type=float)
    parser.add_argument('--mimic_loss_coef', default=20, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    # clip
    parser.add_argument('--ft_clip_with_small_lr', action='store_true',
                        help='Use smaller learning rate to finetune clip weights')
    parser.add_argument('--with_clip_label', action='store_true', help='Use clip to classify HOI')
    parser.add_argument('--with_obj_clip_label', action='store_true', help='Use clip to classify object')
    parser.add_argument('--clip_model', default='ViT-B/32',
                        help='clip pretrained model path')
    parser.add_argument('--fix_clip', action='store_true', help='')
    parser.add_argument('--clip_embed_dim', default=512, type=int)

    # zero/few shot type
    parser.add_argument('--zero_shot_type', default='default',
                        help='default, rare_first, non_rare_first, unseen_object, unseen_verb')
    parser.add_argument('--del_unseen', action='store_true', help='')
    # old parameter
    parser.add_argument('--fix_backbone_mode', nargs='+', default=[], help='fix (part of) backbone')

    # others
    parser.add_argument('--use_ddp', default=1, type=int)
    parser.add_argument('--with_random_shuffle', default=2, type=int, help='Time of random shuffle of annotation')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--opt_sched', default='multiStep', type=str, help='type of scheduler')
    parser.add_argument('--no_clip_cls_init', action='store_true',
                        help='not init classifier weight with clip text encoder')
    parser.add_argument('--enable_amp', action='store_true', help='')
    parser.add_argument('--opt_level', default='O2', help='half precision optimization level', choices=('O1', 'O2'))
    parser.add_argument('--fix_clip_label', action='store_true', help='')
    parser.add_argument('--with_rec_loss', action='store_true', help='')
    parser.add_argument('--rec_loss_coef', default=2, type=float)
    parser.add_argument('--no_training', action='store_true', help='')
    parser.add_argument('--dataset_root', default='GEN', help='')
    parser.add_argument('--model_name', default='GEN', help='')
    parser.add_argument('--eval_location', action='store_true', help='')
    # DAB
    parser.add_argument('--enable_cp', action='store_true',
                        help="use checkpoint to save memory")
    parser.add_argument('--no_fix_clip_linear', action='store_true',
                        help="")
    parser.add_argument('--analysis', action='store_true')

    # tmp args
    parser.add_argument('--alternative', default=1, type=int)
    parser.add_argument('--eval_each_ap', action='store_true')
    parser.add_argument('--topk_hoi', default=10, type=int)
    parser.add_argument('--inter_dec_layers', default=3, type=int)

    # verb setting
    parser.add_argument('--verb_pth', default='', help='location for predefined verb feature', type=str)
    parser.add_argument('--verb_weight', default=0.5, type=float)
    # fractional training
    parser.add_argument('--frac', default=-1., type=float)

    # validation split
    parser.add_argument('--validation_split', default=-1., type=int)
    parser.add_argument('--lr_drop_gamma', default=0.1, type=float)

    # zero shot enhancement
    parser.add_argument('--training_free_enhancement_path', default='', type=str)

    return parser


def main(args):
    if args.use_ddp == 1:
        utils.init_distributed_mode(args)
    else:
        args.distributed = False

    # args.save_points = [int(i) for i in args.save_points]

    print('setting up seeds')
    setup_seed(233)

    # sys.exit(0)

    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    print('****************')
    # print(model)
    print(args.model_name)
    print('****************')

    model_without_ddp = model
    if args.distributed:
        if args.enable_amp:
            raise NotImplementedError
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

    # model = convert_syncbn_model(model)

    for name, p in model.named_parameters():
        if 'eval_visual_projection' in name:
            p.requires_grad = False

    if args.fix_clip:
        for name, p in model.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name or 'clip_model' in name:
                p.requires_grad = False

    if args.no_fix_clip_linear:
        for name, p in model.named_parameters():
            if 'obj_visual_projection' in name or 'visual_projection' in name:
                p.requires_grad = True

    if args.ft_clip_with_small_lr:
        if args.with_obj_clip_label and args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'obj_visual_projection' not in n
                            and 'clip_model' not in n and p.requires_grad and 'T5_model' not in n]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               (
                                       'visual_projection' in n or 'obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               (
                                       'T5_model' in n or 'llm' in n) and p.requires_grad],
                    "lr": args.lr_llm,
                },
            ]
        elif args.with_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        elif args.with_obj_clip_label:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and 'obj_visual_projection' not in n and 'clip_model' not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               ('obj_visual_projection' in n or 'clip_model' in n) and p.requires_grad],
                    "lr": args.lr_clip,
                },
            ]
        else:
            raise
    else:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.opt_sched == 'multiStep':
        lr_scheduler = MultiStepLRWarmup(optimizer, [args.lr_drop], warmup_iter=0, warmup_ratio=0.01,
                                         gamma=args.lr_drop_gamma)
    elif args.opt_sched == 'cosine':
        lr_scheduler = CosineAnnealingLRWarmup(optimizer, verbose=False,
                                               warmup_iter=500,
                                               warmup_ratio=0.01,
                                               T_max=args.epochs - 1,
                                               eta_min=0.01)
    else:
        raise KeyError('Unsupported scheduler type')

    print('init dataloader')
    # train dataloader initialization
    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # test and val dataloader initialization

    test_split = 'val'
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set=test_split, args=args)
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    print('dataloader finished')

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    # init logging
    _LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
    _DATE_FMT = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
    LOGGER = logging.getLogger('__main__')  # this is the global logger
    fh = logging.FileHandler(os.path.join(output_dir, 'training_log.txt'))
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

    if args.resume and os.path.exists(args.resume):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

        # if args.enable_amp:
        #     amp.load_state_dict(checkpoint['amp'])

    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.eval:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    if args.eval:
        if not os.path.exists(output_dir / "log.txt"):
            with open(output_dir / "log.txt", 'w') as f:
                f.write('')
        with open(output_dir / "log.txt", 'r') as f:
            previous_log = f.read()

        if 'Test result:' not in previous_log:
            print('Evaluating in test split!')
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_test,
                                      args.subject_category_id, device, args)

            if args.output_dir and utils.is_main_process():
                #  add eval in log for my convenience
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Test result:' + json.dumps(test_stats) + "\n")
                LOGGER.info('Epoch Test: [{}] '.format('eval') + json.dumps(test_stats))

        if 'Val result:' not in previous_log:
            print('Evaluating in val split!')
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                      args.subject_category_id, device, args)

            if args.output_dir and utils.is_main_process():
                #  add eval in log for my convenience
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Val result:' + json.dumps(test_stats) + "\n")
                LOGGER.info('Epoch Val: [{}] '.format('eval') + json.dumps(test_stats))
        return

    best_performance = 0
    if args.resume and os.path.exists(args.resume):
        try:
            with open(output_dir / "log.txt", 'r') as f:
                previous_log = f.read().split('\n')
            previous_log.remove('')
            test_stats = json.loads(previous_log[-1])
            if args.dataset_file == 'hico':
                performance = test_stats['mAP']
            elif args.dataset_file == 'vcoco':
                performance = test_stats['mAP_all']
            elif args.dataset_file == 'hoia':
                performance = test_stats['mAP']
            best_performance = performance
        except:
            best_performance = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, lr_scheduler,
            args.gradient_accumulation_steps, args.enable_amp, args.no_training, args)

        lr_scheduler.step()
        # if epoch == args.epochs - 1:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
        utils.save_on_master({
                                 'model': model_without_ddp.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'lr_scheduler': lr_scheduler.state_dict(),
                                 # 'amp': amp.state_dict(),
                                 'epoch': epoch,
                                 'args': args,
                             } if args.enable_amp else {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            # 'amp': None,
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val,
                                  args.subject_category_id, device, args)
        if args.dataset_file == 'hico':
            performance = test_stats['mAP']
        elif args.dataset_file == 'vcoco':
            performance = test_stats['mAP_all']
        elif args.dataset_file == 'hoia':
            performance = test_stats['mAP']

        if performance > best_performance:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                                     'model': model_without_ddp.state_dict(),
                                     'optimizer': optimizer.state_dict(),
                                     'lr_scheduler': lr_scheduler.state_dict(),
                                     # 'amp': amp.state_dict(),
                                     'epoch': epoch,
                                     'args': args,
                                 } if args.enable_amp else {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                # 'amp': None,
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

            if epoch in args.save_points and utils.is_main_process():
                checkpoint_path = os.path.join(output_dir, f'best_before_epoch_{epoch}.pth')
                print('achieve save point')
                if os.path.exists(os.path.join(output_dir, 'checkpoint_best.pth')):
                    os.system(f"cp {os.path.join(output_dir, 'checkpoint_best.pth')} {checkpoint_path}")
                elif os.path.exists(os.path.join(output_dir, 'checkpoint_last.pth')):
                    os.system(f"cp {os.path.join(output_dir, 'checkpoint_last.pth')} {checkpoint_path}")
                else:
                    raise ValueError

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            LOGGER.info('Epoch: [{}] '.format(epoch) + json.dumps(log_stats))

            #  add eval in log for my convenience
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")
            LOGGER.info('Epoch: [{}] '.format(epoch) + json.dumps(test_stats))

        if epoch == args.epochs - 1 and os.path.exists(os.path.join(output_dir, 'checkpoint_best.pth')):
            print('Loading best val checkpoint!')
            checkpoint = torch.load(os.path.join(output_dir, 'checkpoint_best.pth'), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = -1
            model.to(device)
            print('Final evaluating in test split!')
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_test,
                                      args.subject_category_id, device, args)

            if args.output_dir and utils.is_main_process():
                #  add eval in log for my convenience
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Test result:' + json.dumps(test_stats) + "\n")
                LOGGER.info('Epoch Test: [{}] '.format(epoch) + json.dumps(test_stats))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GEN VLKT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
