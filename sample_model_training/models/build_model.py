import models
import torch

def build_model(args):
    model_name=args['model_type']
    model = models.__dict__[args.pop('model_type')](**args)
    model.init_weights(**args)
    if args['test_mode']:
        model.eval()
    return model,model_name
