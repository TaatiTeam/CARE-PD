import torch
from torch import nn


def normal_init_(layer, mean_, sd_, bias, norm_bias=True):
    """Intialization of layers with normal distribution with mean and bias"""
    classname = layer.__class__.__name__
    # Only use the convolutional layers of the module
    # if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
    if classname.find('Linear') != -1:
        print('[INFO] (normal_init) Initializing layer {}'.format(classname))
        layer.weight.data.normal_(mean_, sd_)
        if norm_bias:
            layer.bias.data.normal_(bias, 0.05)
        else:
            layer.bias.data.fill_(bias)


def weight_init(
        module,
        mean_=0,
        sd_=0.004,
        bias=0.0,
        norm_bias=False,
        init_fn_=normal_init_):
    """Initialization of layers with normal distribution"""
    moduleclass = module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    init_fn_(l, mean_, sd_, bias, norm_bias)
            else:
                init_fn_(layer, mean_, sd_, bias, norm_bias)
    except TypeError:
        init_fn_(module, mean_, sd_, bias, norm_bias)


def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
        nn.init.xavier_uniform_(layer.weight.data)
        # nninit.xavier_normal(layer.bias.data)
        if norm_bias:
            layer.bias.data.normal_(0, 0.05)
        else:
            layer.bias.data.zero_()
