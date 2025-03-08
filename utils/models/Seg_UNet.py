import torch
import torch.nn as nn

#二重卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//4, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//4, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)



#上采样
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#下采样
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

#Unet
class UNet_base(nn.Module):
    def __init__(self, n_channels, chs=(16, 32, 64, 128, 256, 128, 64, 32, 16),token_weight = 0.1):
        super(UNet_base, self).__init__()
        self.n_channels = n_channels
        self.token_weight = token_weight
        self.inc = DoubleConv(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])
        self.up1 = Up(chs[4] + chs[3], chs[5])
        self.up2 = Up(chs[5] + chs[2], chs[6])
        self.up3 = Up(chs[6] + chs[1], chs[7])
        self.up4 = Up(chs[7] + chs[0], chs[8])
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x, y):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 避免对 y 原地修改
        if y is not None:
            y_weighted = y * self.token_weight
            y1 = self.inc(y_weighted.clone())
            y2 = self.down1(y1)
            y3 = self.down2(y2)
            y4 = self.down3(y3)

        # 确保跳跃连接不会引发问题
            z4 = x4.clone() + y4.clone()  # 防止 x4 或 y4 的修改影响梯度
            z3 = x3.clone() + y3.clone()
            z2 = x2.clone() + y2.clone()
            z1 = x1.clone() + y1.clone()

        else:
            z4 = x4
            z3 = x3
            z2 = x2
            z1 = x1

        x = self.up1(x5, z4)
        x = self.up2(x, z3)
        x = self.up3(x, z2)
        x = self.up4(x, z1)

        return x

#分割网络
class UNet_seg(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, depth=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_seg, self).__init__()
        self.token_weight = 0.1
        self.unet = UNet_base(n_channels=n_channels, chs=depth, token_weight=self.token_weight)
        self.out_conv = nn.Conv2d(depth[-1], n_classes, 1)
    def forward(self, x, y=None):
        x = self.unet(x, y)
        x = self.out_conv(x)
        return x