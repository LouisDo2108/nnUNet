from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
import torch 
import torch.nn as nn
from nnunetv2.tuanluc_dev.utils import *
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.tuanluc_dev.get_network_from_plans import load_acsconv_dict, replace_conv3d_and_load_weight_from_acsconv, read_custom_network_config


class HGGLGGClassifier(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_rate=0.5, return_skips=False, custom_network_config_path=None):
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
        
        # Create a plain nnUNet encoder
        self.encoder = PlainConvEncoder(**encoder_params)
        
        if custom_network_config_path is not None:
            # Read the custom config
            custom_network_config = read_custom_network_config(custom_network_config_path)
            
            # Replace the conv3d with acsconv and load the weights if needed
            if custom_network_config["acsconv"]:
                acsconv_dict = load_acsconv_dict(custom_network_config)
                replace_conv3d_and_load_weight_from_acsconv(self.encoder, custom_network_config, acsconv_dict)
        
        # Add a classifier head
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
    def __init__(self, num_classes, dropout_rate=0.5, return_skips=False, custom_network_config_path=None):
        super(ImageNetBratsClassifier, self).__init__()
        
        if custom_network_config_path is not None:
            # Read the custom config
            custom_network_config = read_custom_network_config(custom_network_config_path)
            
            # Replace the conv3d with acsconv and load the weights if needed
            if custom_network_config["acsconv"]:
                acsconv_dict = load_acsconv_dict(custom_network_config)
                replace_conv3d_and_load_weight_from_acsconv(self.encoder, custom_network_config, acsconv_dict)
        
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


if __name__ == "__main__":
    classifier = HGGLGGClassifier(4, 2, custom_network_config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/base.yaml")
    classifier.load_state_dict(torch.load("/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_55.pt"), strict=False)
    print(classifier)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('dataset_name_or_id', type=str,
    #                     help="Dataset name or ID to train with")
    # parser.add_argument('configuration', type=str,
    #                     help="Configuration that should be trained")
    # parser.add_argument('fold', type=str,
    #                     help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    # parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
    #                     help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    # parser.add_argument('-num_gpus', type=int, default=1, required=False,
    #                     help='Specify the number of GPUs to use for training')
    # parser.add_argument('-custom_cfg_path', type=str, default=None, required=False,
    #                     help='[OPTIONAL] Custom network configuration YAML file path. If not specified, the default is None.')
    # args = parser.parse_args()


    # model = HGGLGGClassifier_v2(4, 2, custom_network_config_path=args.custom_cfg_path)
    # nnunet_trainer = get_trainer_from_args(
    #     args.dataset_name_or_id, args.configuration, args.fold, args.tr,
    #     device=torch.device('cuda'), custom_network_config_path=args.custom_cfg_path
    # )
    # nnunet_trainer.initialize()
    # print(nnunet_trainer.network.encoder)