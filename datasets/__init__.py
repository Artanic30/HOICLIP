from .datasets_gen.hico import build as build_hico_gen
from .datasets_gen.vcoco import build as build_vcoco_gen

from .datasets_generate_feature.hico import build as build_hico_generate_verb
from .datasets_generate_feature.vcoco import build as build_vcoco_generate_verb

def build_dataset(image_set, args):
    if args.dataset_root == "GEN":
        if args.dataset_file == 'hico':
            return build_hico_gen(image_set, args)
        if args.dataset_file == 'vcoco':
            return build_vcoco_gen(image_set, args)
    elif args.dataset_root == "GENERATE_VERB":
        if args.dataset_file == 'hico':
            return build_hico_generate_verb(image_set, args)
        if args.dataset_file == 'vcoco':
            return build_vcoco_generate_verb(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
