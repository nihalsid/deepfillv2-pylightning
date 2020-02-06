import numpy as np
import torch


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


def get_generator(options):
    from . import InpaintSAGenerator
    from . import InpaintUNetGenerator
    relu = None
    if options.no_leaky_relu:
        relu = torch.nn.ReLU(inplace=True)
    else:
        relu = torch.nn.LeakyReLU(0.2, inplace=True)
    if options.gen_type == 'GSAGAN':
        return InpaintSAGenerator.InpaintSAGenerator(options.input_nc, activation=relu, use_gated_convs=(not options.no_gated_conv), no_attention=options.no_attention, no_intermediate_input_filling=options.no_intermediate_input_filling)
    # Leaky relu no effect on UNet
    elif options.gen_type == 'UNet':
        return InpaintUNetGenerator.InpaintUNetGenerator(options.input_nc, options.unet_num_downs)