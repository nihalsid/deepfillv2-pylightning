import torch


class InpaintUNetGenerator(torch.nn.Module):
    """
        Borrowed from Pix2Pix github implementation
    """

    def __init__(self, n_in_channel, num_downs, ngf=64, norm_layer=torch.nn.BatchNorm2d, use_dropout=False):
        super(InpaintUNetGenerator, self).__init__()
        # construct unet structure
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UNetSkipConnectionBlock(3, ngf, input_nc=n_in_channel, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, images, mask):
        """Standard forward"""
        masked_x = images * (1 - mask) + mask
        output = self.model(torch.cat((masked_x, mask), dim=1))
        return output, output


class UNetSkipConnectionBlock(torch.nn.Module):
    """
        Borrowed from pix2pix github repo
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=torch.nn.BatchNorm2d, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == torch.nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = torch.nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = torch.nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, torch.nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc, outer_nc,kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc,kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [torch.nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
