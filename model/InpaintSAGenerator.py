import torch
from model.Layers.GatedConvolutions import GatedConv2dWithActivation, GatedDeConv2dWithActivation, NonGatedConv2dWithActivation, NonGatedDeConv2dWithActivation
from model.Layers.SelfAttention import SelfAttention
from model import get_pad


class InpaintSAGenerator(torch.nn.Module):

    def __init__(self, n_in_channel, activation, use_gated_convs, no_attention, no_intermediate_input_filling):
        super(InpaintSAGenerator, self).__init__()
        self.no_attention = no_attention
        self.no_intermediate_input_filling = no_intermediate_input_filling
        if use_gated_convs:
            conv_2d_with_activation = GatedConv2dWithActivation
            deconv_2d_with_activation = GatedDeConv2dWithActivation
        else:
            conv_2d_with_activation = NonGatedConv2dWithActivation
            deconv_2d_with_activation = NonGatedDeConv2dWithActivation
        cnum = 32
        self.coarse_net = torch.nn.Sequential(
            conv_2d_with_activation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1), activation=activation),

            conv_2d_with_activation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2), activation=activation),
            conv_2d_with_activation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activation),

            conv_2d_with_activation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),

            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),

            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),

            deconv_2d_with_activation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activation),
            conv_2d_with_activation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activation),
            deconv_2d_with_activation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1), activation=activation),

            conv_2d_with_activation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1), activation=activation),
            conv_2d_with_activation(cnum // 2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_net = torch.nn.Sequential(
            conv_2d_with_activation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1), activation=activation),

            conv_2d_with_activation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2), activation=activation),
            conv_2d_with_activation(cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2), activation=activation),

            conv_2d_with_activation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4), activation=activation),

            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16), activation=activation)
        )

        self.refine_attention = torch.nn.Sequential(SelfAttention(4 * cnum), torch.nn.ReLU())

        self.refine_upsample_net = torch.nn.Sequential(
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),
            conv_2d_with_activation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1), activation=activation),
            deconv_2d_with_activation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activation),
            conv_2d_with_activation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1), activation=activation),
            deconv_2d_with_activation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1), activation=activation),
            conv_2d_with_activation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1), activation=activation),
            conv_2d_with_activation(cnum // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )

    def forward(self, images, mask):
        masked_x = images * (1 - mask) + mask
        coarse_x = torch.clamp(self.coarse_net(torch.cat((masked_x, mask), dim=1)), -1., 1.)
        if not self.no_intermediate_input_filling:
            masked_x = images * (1 - mask) + coarse_x * mask
        else:
            masked_x = coarse_x
        x = self.refine_net(torch.cat((masked_x, mask), dim=1))
        if not self.no_attention:
            x = self.refine_attention(x)
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)
        return coarse_x, x
