import torch
import torch.nn as nn
import torch.nn.functional as F



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


class IRDPModule(nn.Module):
    def __init__(self, nIn, alpha,d1,d2,d3,d4, kSize=3):
        super().__init__()
        C_expand = int(alpha * nIn)
        self.conv1x1 = Conv(nIn , C_expand, 1, 1, padding=0, bn_acti=False)
        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(C_expand, C_expand,kSize, 1, padding=1, bn_acti=True)

        self.d1conv3x3 = Conv(C_expand, C_expand, kSize, 1,
                             padding=d1,dilation=d1, groups=C_expand, bn_acti=True)
        self.d2conv3x3 =  Conv(C_expand, C_expand,  kSize, 1,
                             padding=d2,dilation=d2, groups=C_expand, bn_acti=True)
        self.d3conv3x3 =  Conv(C_expand, C_expand,  kSize, 1,
                             padding=d3,dilation=d3, groups=C_expand, bn_acti=True)
        self.d4conv3x3 = Conv(C_expand, C_expand,  kSize, 1,
                             padding=d4,dilation=d4, groups=C_expand, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn)
        self.final_conv3x3 = Conv(C_expand, nIn,kSize, 1, padding=1, bn_acti=True,groups=nIn)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1(output)

        a1 = self.conv3x3(output)
        a2 = self.d1conv3x3(output)
        a3 = self.d3conv3x3(output)
        com_a2_a3=a2+a3
        com_a1_a2=a1+a2
        a4=self.d2conv3x3(com_a1_a2)
        a5=self.d4conv3x3(com_a2_a3)
        com=a4+a5
        output=self.final_conv3x3(com)
        output=output+input
        output=self.bn_relu_2(output)

        return output


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x1 = Conv(nIn, nIn, kSize=(3,1),stride=(1,2), padding=(1,0))
        self.conv1x3 = Conv(nIn, nConv, kSize=(1,3), stride=(2,1), padding=(0,1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x1(input)
        output = self.conv1x3(output)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

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


class IRDPNet(nn.Module):
    def __init__(self,in_channels, classes=19, block_1=2, block_2=5):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(in_channels, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True,groups=32),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True,groups=32),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + in_channels)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + in_channels, 64)
        self.IRDP_Block_1 = nn.Sequential()
        self.IRDP_Block_1.add_module("IRDP_Module_1_" + str(1), IRDPModule(64,1,1,2,1,2))
        for i in range(0, block_1):
            self.IRDP_Block_1.add_module("IRDP_Module_1_" + str(i+1), IRDPModule(64,1,2,4,8,16))
        self.bn_prelu_2 = BNPReLU(128 + in_channels)

        # DAB Block 2
        self.downsample_2 = DownSamplingBlock(128 + in_channels, 128)
        self.IRDP_Block_2= nn.Sequential()
        self.IRDP_Block_2.add_module("IRDP_Module_2_" + str(1), IRDPModule(128,1,1,2,1,2))
        for i in range(0, block_2):
            self.IRDP_Block_2.add_module("IRDP_Module_2_"+ str(i+1),
                                       IRDPModule(128,1,2,4,8,16))
        self.bn_prelu_3 = BNPReLU(256 + in_channels)

        self.classifier = nn.Sequential(Conv(256+in_channels, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # IRDP Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.IRDP_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # IRDP Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.IRDP_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        out = self.classifier(output2_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IRDPNet(in_channels=1,classes=1).to(device)
    # summary(model,(3,512,1024))
    a = torch.rand(size=(1, 1, 512, 512)).to(device)
    print(model(a).shape)