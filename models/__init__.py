from .models_gen.gen_vlkt import build as build_gen
from .models_hoiclip.hoiclip import build as build_models_hoiclip
from .visualization_hoiclip.gen_vlkt import build as visualization
from .generate_image_feature.generate_verb import build as generate_verb


def build_model(args):
    if args.model_name == "HOICLIP":
        return build_models_hoiclip(args)
    elif args.model_name == "GEN":
        return build_gen(args)
    elif args.model_name == "VISUALIZATION":
        return visualization(args)
    elif args.model_name == "GENERATE_VERB":
        return generate_verb(args)

    raise ValueError(f'Model {args.model_name} not supported')
