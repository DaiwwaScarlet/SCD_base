import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from models.CSWin_Transformer import mit



args = {'hidden_size': 128 * 3,
        'mlp_dim': 256 * 3,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.}


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()

        # 反卷积层 上采样操作 高层特征图的空间尺寸放大两倍
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)

        #
        in_channels = in_channels_high + in_channels_low // scale_ratio

        # 调整通道
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low // scale_ratio),
            nn.BatchNorm2d(in_channels_low // scale_ratio),
            nn.ReLU(inplace=True))

        # 解码
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)

        # 高底层 拼接
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


class _DecoderBlock2(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock2, self).__init__()

        # 反卷积层，上采样操作，高层特征图的空间尺寸放大两倍
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)

        # transit 部分的输入和输出通道数不再是缩放的，而是直接基于输入的实际通道数和输出的需求
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, out_channels),  # 修改通道数以确保与输入一致
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 解码部分，结合高层和低层特征
        in_channels = in_channels_high + out_channels
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, low_feat):
        # 上采样高层特征
        x = self.up(x)

        # 调整低层特征通道数以匹配高层特征
        low_feat = self.transit(low_feat)

        # 拼接高层与低层特征
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x



class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):  # 输入通道数，默认3（RGB）
        super(FCN, self).__init__()

        # resnet34 基础模型加载，加载在 ImageNet 上预训练的权重
        resnet = models.resnet34(pretrained=pretrained)

        # 新定义一个卷积层，卷积核7 步长2 填充3
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 权重前 3 个通道直接复制自 ResNet-34 预训练模型的第一层卷积层
        # (64,3,7,7) 64为输出通道数（卷积核个数） 3输入通道数 7，7空间尺寸
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])

        # ResNet 预训练的前三个通道权重来初始化其余通道
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        # 第一层：newconv1+批量归一化+激活函数
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        # 池化
        self.maxpool = resnet.maxpool
        # resnet 四个特征提取层
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        #
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # 自定义头部 512->128 减少通道数但是不改变特征图尺寸 批量归一化 激活函数
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())

        # 初始化权重
        initialize_weights(self.head)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SCanNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=512):
        super(SCanNet, self).__init__()
        feat_size = input_size // 4

        # 编码器
        self.FCN = FCN(in_channels, pretrained=True)

        # 拼接后的单独的二元变化检测编码器
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)

        #
        self.transformer = mit(img_size=feat_size, in_chans=128 * 3, embed_dim=128 * 3)

        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2) #融合了双层通道数要调整为一半
        self.Dec1 = _DecoderBlock(128, 64, 128)
        self.Dec2 = _DecoderBlock(128, 64, 128)

        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD,
                           self.classifierCD)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):

        x = self.FCN.layer0(x)  # size:1/2
        x = self.FCN.maxpool(x)  # size:1/4
        x_low = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x_low)  # size:1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low

    def CD_forward(self, x1, x2):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x2], 1)  # 拼接
        xc = self.resCD(x)
        return x1, x2, xc

    def forward(self, x1, x2):
        x_size = x1.size()

        # 3 512 512  ->  low:64,128,128    high:128,64,64
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)

        # 256, 64, 64  ->  128, 64, 64
        x1, x2, xc = self.CD_forward(x1, x2)

        # 128,64,64 + 64,128,128 -> 128, 128, 128
        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)

        # 64, 128, 128  拼接  128, 128, 128
        xc_low = torch.cat([x1_low, x2_low], 1)

        # 128, 64, 64 + 128, 128, 128 -> 128, 128, 128
        xc = self.DecCD(xc, xc_low)

        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]

        # 128, 128, 128
        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)

        out3 = out1
        return (F.upsample(change, x_size[2:], mode='bilinear'),
                F.upsample(out1, x_size[2:], mode='bilinear'),
                F.upsample(out2, x_size[2:], mode='bilinear'))


class SepNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=512):
        super(SepNet, self).__init__()
        feat_size = input_size // 4

        # 使用 torch.hub 加载 DINOv2 模型
        self.dinov2_vits14 = torch.hub.load('', 'dinov2_vits14', source='local').cuda()

        # 用于调整通道数的 1x1 卷积层
        self.channel_adjust = nn.Conv2d(384, 128, kernel_size=1)

        # 拼接后的单独的二元变化检测编码器
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)

        # Transformer模块
        self.transformer = mit(img_size=feat_size, in_chans=128 * 3, embed_dim=128 * 3)

        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1 = _DecoderBlock(128, 64, 128)
        self.Dec2 = _DecoderBlock(128, 64, 128)

        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD,
                           self.classifierCD)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def backbone(self, x):
        # 使用DINOv2提取特征
        with torch.no_grad():
            features_dict = self.dinov2_vits14.forward_features(x)  # 提取特征
            x_patchtokens = features_dict['x_norm_patchtokens']  # 获取patch tokens

        # 将patch tokens重整为与后续模块兼容的格式
        b, num_patches, feat_dim = x_patchtokens.shape  # 假设x_patchtokens = (batch_size, 1600, 384)
        spatial_dim = int(num_patches ** 0.5)  # 重整到 (b, 384, 40, 40) 如果 num_patches = 1600
        x_patchtokens = x_patchtokens.permute(0, 2, 1).view(b, feat_dim, spatial_dim, spatial_dim)  # (b, 384, 40, 40)

        # 通过1x1卷积调整通道数，以便与解码器兼容
        x_patchtokens = self.channel_adjust(x_patchtokens)  # 转换到 (b, 128, 40, 40)

        # 上采样到 (b, 128, 64, 64)
        x_patchtokens = F.interpolate(x_patchtokens, size=(64, 64), mode='bilinear', align_corners=False)

        # 使用可学习的上采样模块来匹配低层次特征图的大小
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2).cuda()
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2).cuda()

        return x_patchtokens

    def CD_forward(self, x1, x2):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x2], 1)  # 拼接
        xc = self.resCD(x)
        return x1, x2, xc

    def forward(self, x1, x2):
        x_size = x1.size()

        # 确保输入图像的尺寸是 patch 尺寸的整数倍
        new_height = (x1.size(2) // 14) * 14
        new_width = (x1.size(3) // 14) * 14
        if x1.size(2) != new_height or x1.size(3) != new_width:
            x1 = F.interpolate(x1, size=(new_height, new_width), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=(new_height, new_width), mode='bilinear', align_corners=False)

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        x1, x2, xc = self.CD_forward(x1, x2)


        x1_low = self.upsample1(x1)
        x2_low = self.upsample2(x2)

        # 解码阶段
        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)

        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]

        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)

        return (F.upsample(change, x_size[2:], mode='bilinear'),
                F.upsample(out1, x_size[2:], mode='bilinear'),
                F.upsample(out2, x_size[2:], mode='bilinear'))

#
#
# class SepNet(nn.Module):
#     def __init__(self, in_channels=3, num_classes=7, input_size=512):
#         super(SepNet, self).__init__()
#         feat_size = input_size // 4
#
#         # 使用 torch.hub 加载 DINOv2 模型
#         self.dinov2_vits14 = torch.hub.load('', 'dinov2_vits14', source='local').cuda()
#
#         # 用于调整通道数的 1x1 卷积层
#         self.channel_adjust = nn.Conv2d(384, 128, kernel_size=1)
#
#         # 编码器
#         self.FCN = FCN(in_channels, pretrained=True)
#
#         # 拼接后的单独的二元变化检测编码器
#         self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
#
#         #
#         self.transformer = mit(img_size=feat_size, in_chans=128 * 3, embed_dim=128 * 3)
#
#         self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
#         self.Dec1 = _DecoderBlock(128, 64, 128)
#         self.Dec2 = _DecoderBlock(128, 64, 128)
#
#         self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
#         self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
#         self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
#                                           nn.Conv2d(64, 1, kernel_size=1))
#
#         initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD,
#                            self.classifierCD)
#
#
#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 conv1x1(inplanes, planes, stride),
#                 nn.BatchNorm2d(planes))
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def base_forward(self, x):
#
#         x = self.FCN.layer0(x)  # size:1/2
#         x = self.FCN.maxpool(x)  # size:1/4
#         x_low = self.FCN.layer1(x)  # size:1/4
#         x = self.FCN.layer2(x_low)  # size:1/8
#         x = self.FCN.layer3(x)
#         x = self.FCN.layer4(x)
#         x = self.FCN.head(x)
#         return x, x_low
#
#     def backbone(self, x):
#         with torch.no_grad():
#             features_dict = self.dinov2_vits14.forward_features(x)  # 提取特征
#             x_patchtokens = features_dict['x_norm_patchtokens']  # 获取 patch tokens
#
#         # 重整 patch tokens 以兼容后续模块
#         b, num_patches, feat_dim = x_patchtokens.shape
#         spatial_dim = int(num_patches ** 0.5)
#         x_patchtokens = x_patchtokens.permute(0, 2, 1).view(b, feat_dim, spatial_dim, spatial_dim)
#
#         # 通道调整
#         x_patchtokens = self.channel_adjust(x_patchtokens)
#         return x_patchtokens
#
#
#     def CD_forward(self, x1, x2):
#         b, c, h, w = x1.size()
#         x = torch.cat([x1, x2], 1)  # 拼接
#         xc = self.resCD(x)
#         return x1, x2, xc
#
#     def forward(self, x1, x2):
#         x_size = x1.size()
#         x1 = self.backbone(x1)
#         x2 = self.backbone(x2)
#         x1, x2, xc = self.CD_forward(x1, x2)
#
#         x = torch.cat([x1, x2, xc], 1)
#         x = self.transformer(x)
#         x1 = x[:, 0:128, :, :]
#         x2 = x[:, 128:256, :, :]
#         xc = x[:, 256:, :, :]
#
#         out1 = self.classifierA(x1)
#         out2 = self.classifierB(x2)
#         change = self.classifierCD(xc)
#
#         return (F.upsample(change, x_size[2:], mode='bilinear'),
#                 F.upsample(out1, x_size[2:], mode='bilinear'),
#                 F.upsample(out2, x_size[2:], mode='bilinear'))
