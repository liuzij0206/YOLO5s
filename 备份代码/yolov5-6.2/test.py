import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

# -------------------------------------AC-FPN------------------------------------------
class CxAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CxAM, self).__init__()
        self.key_conv = nn.Conv2d(in_channels, out_channels//reduction, 1)
        self.query_conv = nn.Conv2d(in_channels, out_channels//reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()    # b, 512, 20, 20, 

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)   # B ,C, W, H --> B, C', N --> B x N x C'

        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # B x C' x N

        R = torch.bmm(proj_query, proj_key).view(m_batchsize, width*height, width, height)  # B, N, N --> B x N x W x H
        # 先进行全局平均池化, 此时 R 的shape为 B x N x 1 x 1, 再进行view, R 的shape为 B x 1 x W x H
        attention_R = self.sigmoid(self.avg(R).view(m_batchsize, -1, width, height))    # B x 1 x W x H

        proj_value = self.value_conv(x) # B x C x W x H

        out = proj_value * attention_R  # B x W x H

        return out


class CnAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(CnAM, self).__init__()
        # 原文中对应的P, Z, S
        self.Z_conv = nn.Conv2d(in_channels*2, out_channels // reduction, 1)
        self.P_conv = nn.Conv2d(in_channels*2, out_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)

    # CnAM使用了FPN中的F5和CEM输出的特征图F
    def forward(self, F5, F):
        m_batchsize, C, width, height = F5.size()   # B, C, W, H 

        proj_query = self.P_conv(F5).view(m_batchsize, -1, width*height).permute(0, 2, 1)  # B x N x C''

        proj_key = self.Z_conv(F5).view(m_batchsize, -1, width * height)  # B x C'' x N

        S = torch.bmm(proj_query, proj_key).view(m_batchsize, width * height, width, height)  # B x N x W x H
        attention_S = self.sigmoid(self.avg(S).view(m_batchsize, -1, width, height))  # B x 1 x W x H

        proj_value = self.value_conv(F)

        out = proj_value * attention_S  # B x W x H

        return out

class DenseBlock(nn.Module):
    #                   输入通道，中间通道，输出通道，扩张操作
    def __init__(self, input_num, num1, num2, rate, drop_out):  
        super(DenseBlock, self).__init__()

        # C: 2048 --> 512 --> 256
        self.conv1x1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=num1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dilaconv = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3, padding=1 * rate, dilation=rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.ConvGN(self.conv1x1(x))
        x = self.relu1(x)
        x = self.dilaconv(x)
        x = self.relu2(x)
        x = self.drop(x)
        return x


class DenseAPP(nn.Module):
    def __init__(self, num_channels=1024):  # in_channels = 2048
        super(DenseAPP, self).__init__()
        self.drop_out = 0.1
        self.channels1 = 512
        self.channels2 = 256
        self.num_channels = num_channels
        self.aspp3 = DenseBlock(self.num_channels, num1=self.channels1, num2=self.channels2, rate=3,
                                drop_out=self.drop_out)
        self.aspp6 = DenseBlock(self.num_channels + self.channels2 * 1, num1=self.channels1, num2=self.channels2,
                                rate=6,
                                drop_out=self.drop_out)
        self.aspp12 = DenseBlock(self.num_channels + self.channels2 * 2, num1=self.channels1, num2=self.channels2,
                                 rate=12,
                                 drop_out=self.drop_out)
        self.aspp18 = DenseBlock(self.num_channels + self.channels2 * 3, num1=self.channels1, num2=self.channels2,
                                 rate=18,
                                 drop_out=self.drop_out)
        self.aspp24 = DenseBlock(self.num_channels + self.channels2 * 4, num1=self.channels1, num2=self.channels2,
                                 rate=24,
                                 drop_out=self.drop_out)
        self.conv1x1 = nn.Conv2d(in_channels=5*self.channels2, out_channels=512, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=512)

    def forward(self, feature):
        aspp3 = self.aspp3(feature)
        feature = torch.concat((aspp3, feature), dim=1)
        aspp6 = self.aspp6(feature)
        feature = torch.concat((aspp6, feature), dim=1)
        aspp12 = self.aspp12(feature)
        feature = torch.concat((aspp12, feature), dim=1)
        aspp18 = self.aspp18(feature)
        feature = torch.concat((aspp18, feature), dim=1)
        aspp24 = self.aspp24(feature)

        x = torch.concat((aspp3, aspp6, aspp12, aspp18, aspp24), dim=1)
        out = self.ConvGN(self.conv1x1(x))  # out_channels = 256
        return out
    
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, c1, c2, extra_blocks=None):
        super().__init__()
        self.dense = DenseAPP(num_channels=1024)        
        # --------增加AM模块，若不想使用，可直接注释掉--------#
        self.CxAM = CxAM(in_channels=512, out_channels=512)
        self.CnAM = CnAM(in_channels=512, out_channels=512) 
        self.conv1x1 = nn.Conv2d(in_channels=c1, out_channels=1024, kernel_size=1)
        # -------------------------------------------------#
        
    def forward(self, x):
        # B, W, H, C --> B, C, W, H
        # x = x.permute(0, 3, 1, 2)
        a = self.conv1x1(x)
        print(x.shape)
        y = self.dense(a)    
        print(y.shape)    
        cxam = self.CxAM(y)
        cnam = self.CnAM(a, y)
        out = cxam + cnam
        print(out.shape)
        # x = self.conv1x1(x)       
        out = out + x
        # B, W, H, C --> B, C, W, H
        # out = out.permute(0, 2, 3, 1)
        return out         
    
    
# a = torch.randn(16, 512, 20, 20)    
# f1 = FeaturePyramidNetwork(512, 512)
# # f2 = DenseAPP(1024)
# # cxam = CxAM(in_channels=512, out_channels=512)
# # cnam = CnAM(in_channels=512, out_channels=512) 
# # c = f2(a)
# # print(c.shape)

# # # print(c)
# # d = cxam(c)
# # print(d.shape)

# # e = cnam(a, c)
# # print(e.shape)

# # de = d + e
# # print(de.shape)

# f = f1(a)
# print(f.shape)

#---------------------------------------------------C3STR----------------------------------------------
from models.SwinTransformer import SwinTransformerLayer
from models.common import C3
class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x

class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
#---------------------------------------------------C3STR----------------------------------------------
c3 = C3STR(512, 1024)
a = torch.randn(16, 512, 20, 20)
a = c3(a) 
print(a.shape)