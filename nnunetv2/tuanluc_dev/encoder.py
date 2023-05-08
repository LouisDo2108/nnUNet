from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.utilities.network_initialization import InitWeights_He
import torch 
import torch.nn as nn
from nnunetv2.tuanluc_dev.utils import *


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class HGGLGGClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_rate=0.5, return_skips=False):
        super(HGGLGGClassifier, self).__init__()
        encoder_params = {
            'input_channels': input_channels,
            'n_stages': 6,
            'features_per_stage': [32, 64, 128, 256, 320, 320],
            'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
            'conv_op': torch.nn.modules.conv.Conv3d,
            'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            'conv_bias': True,
            'norm_op': torch.nn.modules.instancenorm.InstanceNorm3d,
            'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': torch.nn.modules.activation.LeakyReLU,
            'nonlin_kwargs': {'inplace': True},
            'return_skips': return_skips
        }
        self.encoder = PlainConvEncoder(**encoder_params)
        self.encoder.apply(InitWeights_He(1e-2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes-1),
        )
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
        

class ImageNetBratsClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, return_skips=False):
        super(ImageNetBratsClassifier, self).__init__()
        self.encoder, _ = get_model_and_transform("resnet18", pretrained=True)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes-1),
        )
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


# if __name__ == "__main__":
    
#     classifier = HGGLGGClassifier(4, 2)