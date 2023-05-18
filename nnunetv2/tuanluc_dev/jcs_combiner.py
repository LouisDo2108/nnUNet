import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import BatchNorm2d
from nnunetv2.tuanluc_dev.acsconv.operators import ACSConv
from nnunetv2.tuanluc_dev.encoder import HGGLGGClassifier
from nnunetv2.tuanluc_dev.get_network_from_plans import get_network_from_plans
from nnunetv2.tuanluc_dev.get_network_from_plans_dev import get_encoder
from nnunetv2.tuanluc_dev.utils import *

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        # scale = scale.reshape(1, -1, 1, 1)
        # bias = bias.reshape(1, -1, 1, 1)
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ConvBNReLU, self).__init__()
        # self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
        #                       dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv3d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                # self.bn = BatchNorm2d(nOut)
                self.bn = nn.BatchNorm3d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, in_channels//reduction, bias=True)
        self.linear2 = nn.Linear(in_channels//reduction, in_channels)
        self.act = nn.ReLU(inplace=True)
        

    def forward(self, x):
        # N, C, H, W = x.shape
        N, C, D, H, W = x.shape
        # embedding = x.mean(dim=2).mean(dim=2)
        embedding = x.mean(dim=2).mean(dim=2).mean(dim=2)
        fc1 = self.act(self.linear1(embedding))
        fc2 = torch.sigmoid(self.linear2(fc1))
        return x * fc2.view(N, C, 1, 1, 1)


class FuseNet(nn.Module):
    def __init__(self, c1=[1,2,3,4,5], c2=[1,2,3,4,5], out_channels=[1,2,3,4,5]):
        super(FuseNet, self).__init__()
        self.cat_modules = nn.ModuleList()
        self.se_modules = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        for i in range(len(c1)):
            self.cat_modules.append(ConvBNReLU(c1[i]+c2[i], out_channels[i]))
            self.se_modules.append(ConvBNReLU(out_channels[i], out_channels[i]))
            self.fuse_modules.append(ConvBNReLU(out_channels[i], out_channels[i]))

    def forward(self, x1, x2):
        x_new = []
        for i in range(6):
            x1[i] = F.interpolate(x1[i], x2[i].shape[2:], mode='trilinear', align_corners=False)
            m = self.cat_modules[i](torch.cat([x1[i], x2[i]], dim=1))
            #print(m.shape)
            m = self.se_modules[i](m)
            #print(m.shape)
            m = self.fuse_modules[i](m)
            #print(m.shape)
            x_new.append(m)
        return x_new[0], x_new[1], x_new[2], x_new[3], x_new[4], x_new[5]


class FuseModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.se1 = SEBlock(in_channels, in_channels)
        self.conv1 = ConvBNReLU(in_channels, in_channels, use_bn=False)
        self.se2 = SEBlock(in_channels * 3 // 2, in_channels * 3 // 2)
        # self.se2 = SEBlock(in_channels * 3, in_channels * 3)
        self.reduce_conv = ConvBNReLU(in_channels * 3 // 2, in_channels, ksize=1, pad=0, use_bn=False)
        self.conv2 = ConvBNReLU(in_channels, in_channels, use_bn=False)

    def forward(self, low, high):
        x = self.se1(torch.cat([low, high], dim=1))
        x = self.conv1(x)
        x = self.se2(torch.cat([x, high], dim=1))
        x = self.reduce_conv(x)
        x = self.conv2(x)
        return x
    


if __name__ == "__main__":
    # Load the classifier
    classifier = HGGLGGClassifier(4, 2, return_skips=True, custom_network_config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/base.yaml")
    classifier.load_state_dict(torch.load("/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_55.pt"), strict=False)
    classifier.eval()
    classifier_encoder = classifier.encoder

    # Process input using the classifier
    temp = torch.randn(1, 4, 128, 128, 128)
    classifier_out = classifier_encoder(temp)

    # Load the segmenter
    nnunet_trainer = get_encoder()
    model = nnunet_trainer.network
    model = model.to(torch.device("cpu"))
    model.load_state_dict(torch.load("/tmp/htluc/nnunet/nnUNet_results/Dataset032_BraTS2018/nnUNetTrainer__nnUNetPlans__3d_fullres_bs4_hgg_lgg/fold_0/checkpoint_best.pth"), strict=False)
    
    segmenter_encoder = model.encoder
    segmenter_out = segmenter_encoder(temp)
    
    fuse = nn.ModuleList()
    for idx, i in enumerate([32, 64, 128, 256, 320, 320]):
        fuse.append(FuseModule(64))
        print(fuse[idx](classifier_out[idx], segmenter_out[idx]).shape)
        print(count_parameters(fuse[idx]))
        break
    
    
    
    # fuse = FuseNet(c1=[32, 64, 128, 256, 320, 320], c2=[32, 64, 128, 256, 320, 320], out_channels=[32, 64, 128, 256, 320, 320])
    # result = fuse(classifier_out, segmenter_out)
    # for i in result:  
    #     print(i.shape)
    