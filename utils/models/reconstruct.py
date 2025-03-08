import torch
import torch.nn as nn


# 定义卷积和池化层
class reconstruct(nn.Module):
    def __init__(self):
        super(reconstruct, self).__init__()

        # 使用卷积层和池化层逐步减少空间维度
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        #池化层缩小尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 转置卷积，将通道数和空间尺寸同时调整
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        # 最后精确调整到目标尺寸
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv1(x)  # (1, 592, 4096) -> (32, 296, 2048)
        x = self.conv2(x)  # (32, 296, 2048) -> (64, 148, 1024)
        x = self.conv3(x)  # (64, 148, 1024) -> (128, 74, 512)
        x = self.conv4(x)  # (128, 74, 512) -> (256, 37, 256)

        x = self.maxpool(x)  # (256, 37, 256) -> (256, 18, 128)
        x = self.maxpool(x)  # (256, 18, 128) -> (256, 9, 64)
        x = self.avgpool(x)  # (256, 9, 64) -> (256, 4, 32)

        x = self.deconv1(x)  # (256, 4, 32) -> (128, 8, 64)
        x = self.deconv2(x)  # (128, 8, 64) -> (64, 16, 128)
        x = self.deconv3(x)  # (64, 16, 128) -> (1, 32, 256)
        x = self.upsample(x)  # (1, 32, 256) -> (1, 224, 224)

        return x

# 示例输入
x = torch.randn(1, 592, 4096)  # 输入大小 (1, 592, 4096)
# 添加额外的维度（通过 unsqueeze）
x = x.unsqueeze(0)  # 变为 (1, 1, 592, 4096) 形式，添加频道维度
# 定义模型并传入输入
model = reconstruct()
output = model(x)
# 打印输出形状
print(output.shape)
