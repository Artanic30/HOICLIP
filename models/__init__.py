from .models_gen.gen_vlkt import build as build_gen
from .models_hoiclip.gen_vlkt import build as build_models_hoiclip
from .visualization_hoiclip.gen_vlkt import build as visualization


def build_model(args):
    if args.model_name == "HOICLIP":
        return build_models_hoiclip(args)
    elif args.model_name == "GEN":
        return build_gen(args)
    elif args.model_name == "VISUALIZATION":
        return visualization(args)

    raise ValueError(f'Model {args.model_name} not supported')
