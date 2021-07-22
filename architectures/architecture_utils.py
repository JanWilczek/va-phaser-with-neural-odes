import torch.nn as nn


def get_nonlinearity(args):
    return getattr(nn, args.nonlinearity)()
