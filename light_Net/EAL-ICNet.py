import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# __all__ = ["DABNet"]

from model.unet_parts_att_transformer import ScaledDotProductAttention, PAM_Module
from segmentation.light_Net.PVT import to_3d, AttentionTSSA, to_4d


# ScaledDotProductAttention
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)

        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)

        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)

        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
    #branch1
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
    #branch2
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2

        output = self.bn_relu_2(output)

        output = self.conv1x1(output)

        return output + input

#
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)  #concate of a 3 × 3 convolution with stride 2 and a 2 × 2 max-pooling

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input

#nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False
# def cnn_to_transformer(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
#     """
#     将 CNN 特征 (B, C, H, W) 转换为 Transformer 输入格式 (B, H*W, C)
#     同时返回 H, W 以便后续还原。
#     """
#     B, C, H, W = x.shape
#     x = x.flatten(2).transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
#     return x, H, W
# def transformer_to_cnn(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#     """
#     将 Transformer 输出 (B, H*W, C) 转换回 CNN 格式 (B, C, H, W)
#     """
#     B, N, C = x.shape
#     assert N == H * W, f"Error: input tokens {N} != H*W ({H}*{W})"
#     x = x.transpose(1, 2).reshape(B, C, H, W)
#     return x

import torch

def cnn_to_transformer(x: torch.Tensor):
    """
    x: [B, C, H, W] -> returns (x_tokens, H, W)
    x_tokens: [B, H*W, C]
    """
    B, C, H, W = x.shape
    # flatten spatial dims then move channel to last dim
    x_tokens = x.flatten(2).transpose(1, 2)   # [B, C, H*W] -> [B, H*W, C]
    return x_tokens, H, W

def transformer_to_cnn(x_tokens: torch.Tensor, H: int, W: int):
    """
    x_tokens: [B, H*W, C] -> returns x: [B, C, H, W]
    """
    B, N, C = x_tokens.shape
    assert N == H * W, f"Token number {N} mismatches H*W ({H}*{W})"
    x = x_tokens.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
    return x

class DABNet(nn.Module):
    def __init__(self,in_channels, classes=19, block_1=3, block_2=6):  #19个分类
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(in_channels, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times 1/2
        self.down_2 = InputInjection(2)  # down-sample the image 2 times 1/2^2=1/4
        self.down_3 = InputInjection(3)  # down-sample the image 3 times 1/2^3=1/8

        self.bn_prelu_1 = BNPReLU(32 + in_channels)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + in_channels, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + in_channels)

        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + in_channels, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + in_channels)

        self.classifier = nn.Sequential(Conv(256+in_channels, classes, 1, 1, padding=0))
        self.AttentionTSSA=AttentionTSSA(dim=128)
        self.PAM=PAM_Module(128)



    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        # Downsample1
        output1_0 = self.downsample_1(output0_cat)

        # DAB Block 1
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # Downsample2
        output2_0 = self.downsample_2(output1_cat)
        output2_0_pam=self.PAM(output2_0 )

        # DAB Block 2
        output2 = self.DAB_Block_2(output2_0)
        # print(output2_0.shape)

        output2_0_3d = to_3d(output2_0)

        # 初始化 AttentionTSSA 模型
        output_3d = self.AttentionTSSA(output2_0_3d)
        output_4d = to_4d(output_3d, 64, 64)
        # output2 = output2 + output2_0_pam  +output_4d+output2_0_pam+output_4d

        output2 = output2+output2_0_pam+output_4d

        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        out = self.classifier(output2_cat)
        # print(out.shape)

        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DABNet(in_channels=1,classes=1).to(device)
    # summary(model,(3,512,1024))
    a = torch.rand(size=(4, 1, 512, 512)).to(device)
    print(model(a).shape)

