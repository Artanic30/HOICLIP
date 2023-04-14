import torch.utils.data
import torchvision

from .datasets_gen.hico import build as build_hico_gen
from .datasets_gen.vcoco import build as build_vcoco_gen


def build_dataset(image_set, args):
    if args.dataset_root == "GEN":
        if args.dataset_file == 'hico':
            return build_hico_gen(image_set, args)
        if args.dataset_file == 'vcoco':
            return build_vcoco_gen(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
