
import torch
import torch.nn as nn

class Net(nn.Module):
    """ 
    A base class provides a common weight initialization scheme.
    """

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if 'conv' in classname.lower():
                nn.init.normal_(m.weight, 0.0, 0.02)

            if 'batchnorm' in classname.lower() \
                or 'groupnorm' in classname.lower():
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

            if 'linear' in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x

##############################
#           U-NET
##############################

class UNetDown(Net):
    def __init__(self, in_size, out_size, normalize=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=True)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(Net):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input):
        x = self.conv(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(Net):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        dropout=0.5
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down4_drop = nn.Dropout(dropout)
        self.down5 = UNetDown(512, 512)
        self.down5_drop = nn.Dropout(dropout)
        self.down6 = UNetDown(512, 512)
        self.down6_drop = nn.Dropout(dropout)
        self.down7 = UNetDown(512, 512)
        self.down7_drop = nn.Dropout(dropout)
        self.down8 = UNetDown(512, 512, normalize=False)
        self.down8_drop = nn.Dropout(dropout)

        self.up1 = UNetUp(512, 512)
        self.up1_drop = nn.Dropout(dropout)
        self.up2 = UNetUp(1024, 512)
        self.up2_drop = nn.Dropout(dropout)
        self.up3 = UNetUp(1024, 512)
        self.up3_drop = nn.Dropout(dropout)
        self.up4 = UNetUp(1024, 512)
        self.up4_drop = nn.Dropout(dropout)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final_rgb = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )


        self.weights_init()

    def forward(self, x):
        ####
        def upsample(x):
            return nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4_drop(self.down4(d3))
        d5 = self.down5_drop(self.down5(d4))
        d6 = self.down6_drop(self.down6(d5))
        d7 = self.down7_drop(self.down7(d6))
        d8 = self.down8_drop(self.down8(d7))
        u1 = self.up1_drop(self.up1(d8, d7))
        u2 = self.up2_drop(self.up2(u1, d6))
        u3 = self.up3_drop(self.up3(u2, d5))
        u4 = self.up4_drop(self.up4(u3, d4))
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u7 = upsample(u7)
        out_rgb = self.final_rgb(u7)

        return out_rgb


##############################
#        Discriminator
##############################

class Discriminator(Net):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers.extend(discriminator_block(in_channels,  64, 2))
        layers.extend(discriminator_block( 64, 128, 2))
        layers.extend(discriminator_block(128, 256, 2))
        layers.extend(discriminator_block(256, 512, 2))
        layers.extend([nn.Conv2d(512, 1, 3, 1, 1)])

        self.model = nn.Sequential(*layers)
        self.weights_init()

    def forward(self, img_AB):
        """
        Image A concatenate to B to condition it
        """
        # Concatenate image and condition image by channels to produce input
        return self.model(img_AB)
