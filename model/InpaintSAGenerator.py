import torch
from model.Layers.GatedConvolutions import GatedConv2dWithActivation, GatedDeConv2dWithActivation
from model.Layers.SelfAttention import SelfAttention
from model import get_pad


class InpaintSAGenerator(torch.nn.Module):

    def __init__(self, n_in_channel):
        super(InpaintSAGenerator, self).__init__()
        cnum = 32
        self.coarse_net = torch.nn.Sequential(
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),

            GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),

            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 4, 2, padding=get_pad(128, 4, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),

            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),

            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None)
        )

        self.refine_net = torch.nn.Sequential(
            GatedConv2dWithActivation(n_in_channel, cnum, 5, 1, padding=get_pad(256, 5, 1)),

            GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2)),
            GatedConv2dWithActivation(cnum, 2 * cnum, 4, 2, padding=get_pad(128, 4, 2)),

            GatedConv2dWithActivation(2 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),

            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        )

        self.refine_attention = torch.nn.Sequential(SelfAttention(4 * cnum), torch.nn.ReLU())

        self.refine_upsample_net = torch.nn.Sequential(
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * cnum, 4 * cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedDeConv2dWithActivation(2, 4 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2 * cnum, 2 * cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedDeConv2dWithActivation(2, 2 * cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum, cnum // 2, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum // 2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None),
        )

    def forward(self, images, mask):
        masked_x = images * (1 - mask) + mask
        coarse_x = torch.clamp(self.coarse_net(torch.cat((masked_x, mask), dim=1)), -1., 1.)
        masked_x = images * (1 - mask) + coarse_x * mask
        x = self.refine_net(torch.cat((masked_x, mask), dim=1))
        x = self.refine_attention(x)
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)
        return coarse_x, x
