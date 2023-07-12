import torch.nn as nn

class ISLESSegNet(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(ISLESSegNet, self).__init__()
        kernel_size = 3
        stride = 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(4,8,kernel_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=stride, padding=padding)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=stride, padding=padding)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=stride, padding=padding)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=stride, padding=padding)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2)
            nn.Upsample((28,28))
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32,16,kernel_size),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2)
            nn.Upsample((56,56))
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(16,8,kernel_size),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2)
            nn.Upsample((112,112))
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(8,2,kernel_size),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2)
            nn.Upsample((224,224))
        )

        #print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print("number of parameters: %.2fM" % (self.get_num_params()))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x.float()

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


""" Parts of the U-Net model """
import torch
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        #self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        #self.down4 = (Down(512, 1024 // factor))
        #self.up1 = (Up(1024, 512 // factor, bilinear))
        #self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up3(x3, x2) # added
        #x = self.up2(x, x3)
        #x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

class CustomTinyVGG(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.classifier = nn.Sequential(
            DoubleConv(in_channels, mid_channels),
            nn.MaxPool2d(2),
            DoubleConv(mid_channels, mid_channels),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=mid_channels, out_features=out_channels)
        )

    def forward(self, x):
        return self.classifier(x)

class CustomTinyVGGLazy(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.classifier = nn.Sequential(
            DoubleConv(in_channels, mid_channels),
            nn.MaxPool2d(2),
            DoubleConv(mid_channels, mid_channels),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(out_features=out_channels)
        )

    def forward(self, x):
        return self.classifier(x)