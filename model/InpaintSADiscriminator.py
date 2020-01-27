import torch
from model.Layers.SelfAttention import SelfAttention
from model import get_pad
from model.Layers.SNConvolution import SNConvWithActivation


class InpaintSADiscriminator(torch.nn.Module):

    def __init__(self, in_channels):
        super(InpaintSADiscriminator, self).__init__()
        cnum = 32

        self.discriminator_net = torch.nn.Sequential(
            SNConvWithActivation(in_channels, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            SelfAttention(8*cnum),
            torch.nn.ReLU(),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(4, 5, 2)),
        )
        # self.linear = torch.nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0), -1))
        return x

