import torch
from torch import nn
import timm

from collections import defaultdict
from natsort import natsorted
from pathlib import Path
from pprint import pprint

import nnunetv2
from nnunetv2.tuanluc_dev.utils import *
from nnunetv2.tuanluc_dev.acsconv.operators import ACSConv
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks

from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from nnunetv2.tuanluc_dev.jcs_combiner import JCSCombiner
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
import torch 
import torch.nn as nn
from nnunetv2.tuanluc_dev.utils import *


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


class JCSConvUnet(PlainConvUNet):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage,
            num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, 
            dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, nonlin_first
        )
        classifier = HGGLGGClassifier(4, 2, return_skips=True, custom_network_config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/base.yaml")
        classifier.load_state_dict(torch.load("/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_55.pt"), strict=False)
        self.classifier = classifier.encoder
        self.classifier.to("cuda")
        self.classifier.eval()
        self.fuse_module_list = nn.ModuleList()
        for x in [32, 64, 128, 256, 320, 320]:
            self.fuse_module_list.append(JCSCombiner(x))
    
    def forward(self, x):
        classifier_out = self.classifier(x)
        skips = self.encoder(x)
        combine = list(zip(classifier_out, skips))
        skips = [self.fuse_module_list[i](cls, seg) for i, (cls, seg) in enumerate(combine)]
        return self.decoder(skips)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or \
            isinstance(module, nn.Conv2d) or \
            isinstance(module, nn.ConvTranspose2d) or \
            isinstance(module, nn.ConvTranspose3d) or \
            isinstance(module, ACSConv):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def read_custom_network_config(custom_network_config_path):
    print("---------- Custom network config ----------")
    try:
        print("Config file name:", Path(custom_network_config_path).name)
        custom_network_config = load_yaml_config_file(custom_network_config_path)
        pprint(custom_network_config)
    except Exception as e:
        print(e)
    print("--------------------------------------------------\n")
    return custom_network_config


def init_weights_from_pretrained_proxy_task_encoder(nnunet_model, custom_network_config_path):
    
    custom_network_config = load_yaml_config_file(custom_network_config_path)
    
    try:
        proxy_encoder_class = custom_network_config["proxy_encoder_class"]    
        proxy_task_encoder_class = recursive_find_python_class(join(nnunetv2.__path__[0], "tuanluc_dev"), proxy_encoder_class, "nnunetv2.tuanluc_dev")
    except Exception as e:
        print("No such proxy encoder class:", proxy_encoder_class)
        raise e
    
    try: 
        proxy_encoder_pretrained_path = custom_network_config["proxy_encoder_pretrained"]
        if proxy_encoder_class == "HGGLGGClassifier":
            pretrained_model = proxy_task_encoder_class(4, 2, return_skips=True)
        loaded = torch.load(proxy_encoder_pretrained_path, map_location=torch.device('cpu'))
        pretrained_model.load_state_dict(loaded, strict=False)
    except Exception as e:
        print("No such pretrained proxy encoder path:", proxy_encoder_class)
        raise e
    
    del loaded
    pretrained_encoder = pretrained_model.encoder
    nnunet_model.encoder = pretrained_encoder.to(torch.device('cuda'))
    
    return nnunet_model, proxy_encoder_pretrained_path


def replace_nnunet_conv3d_with_acsconv_random(model, target_layer):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_nnunet_conv3d_with_acsconv_random(module, target_layer)
            
        if isinstance(module, target_layer):
            setattr(model, n, ACSConv(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups))


def load_acsconv_dict(custom_network_config):
    acs_pretrained = custom_network_config["acs_pretrained"]
    acsconv_dict = None
        
    try:
        print("Trying to load ACSConv pretrained weights as a dictionary")
        if acs_pretrained is None:
            # Random init, do nothing
            pass
        elif Path(acs_pretrained).is_file():
            # Use custom pretrained encoder
            # Check if acs_pretrained is a path to a pretrained encoder
            # Currently only supports "load_resnet18_custom_encoder"
            acs_pretrained = load_resnet18_custom_encoder(acs_pretrained)
            _, acsconv_dict = get_acs_pretrained_weights(model_name=acs_pretrained)
        else:
            _, acsconv_dict = get_acs_pretrained_weights(model_name=acs_pretrained)
        print("Successfully loaded ACSConv pretrained weights as a dictionary\n")
    except Exception as e:
        print("Failed to load ACSConv pretrained weights as a dictionary")
        print("Error: ", e)
        raise e
    return acsconv_dict


def replace_conv3d_and_load_weight_from_acsconv(model, custom_network_config, acsconv_dict):
    try:
        print("Trying to replace Conv3D with ACSConv")
            
        # We always load the random init first
        replace_nnunet_conv3d_with_acsconv_random(model, nn.Conv3d)
        print("Successfully replaced conv3d with random ACSConv")
            
        # Always init with nnUNet init for non-pretrained layers
        # After replacing conv3d with ACSConv
        # Otherwise, these layers are already init with ACSConv default init
        if custom_network_config["nnUNet_init"]:
            model.apply(InitWeights_He(1e-2))
            
        # If we do not have pretrained weights, we are done
        if acsconv_dict is None:
            if custom_network_config["nnUNet_init"]:
                print("Successfully init with nnUNet init")
            else:
                print("Successfully init with ACSConv default init")
            print("--------------------------------------------------\n")
            return model
            
        # From this point, we have pretrained weights that need further processing
        # We have 2 version of initializing a model with ACSConv pretrained weights
        # 1. Replace conv3d with ACSConv with the same in_channels, out_channels, kernel_size, stride
        # 2. Replace conv3d with ACSConv regardlessly (aka, replace "All")
        if custom_network_config["replace_all"]:
            replace_nnunet_encoder_conv3d_with_acs_pretrained_all(model, acsconv_dict)
            print("Successfully load pretrained weights for all ACSConv (All)")
        else:
            replace_nnunet_encoder_conv3d_with_acs_pretrained(model, acsconv_dict)
            print("Successfully load pretrained weights for some ACSConv")
    except Exception as e:
        print("Failed to replace Conv3D with ACSConv")
        print("Error: ", e)
        raise e


def load_resnet18_custom_encoder(pretrained_model_path):
    from nnunetv2.tuanluc_dev.encoder import ImageNetBratsClassifier
    model = ImageNetBratsClassifier(num_classes=2)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')), strict=False)
    print("Loaded pretrained model from {}".format(pretrained_model_path))
    return model.encoder


def get_acs_pretrained_weights(model_name='resnet18', pretrained=True):
    """
    Generate dictionaries of Conv2d and ACSConv layers for a given pre-trained model.

    Args:
        model_name (str): the name of the model to load from the timm library (default: 'resnet18')

    Returns:
        Tuple[Dict[str, List[nn.Conv2d]], Dict[str, List[ACSConv]]]: a tuple of dictionaries
            mapping from layer keys to lists of corresponding Conv2d and ACSConv layers
    """
    try:
        if isinstance(model_name, str):
            model = timm.create_model(model_name, pretrained=pretrained)
            print(f'Loaded model {model_name} from timm library')
        else:
            model = model_name
            print('Loaded model from input')
    except Exception as e:
        print('Error loading model')
        raise e
    
    conv2d_dict = defaultdict(list)
    acsconv_dict = defaultdict(list)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and len(name.split('.')) == 3 and module.kernel_size[0] == 3:
            # parse out the input and output channels, layer number, and index
            layer_name = name.split('.')[0]
            conv_params = module.weight.shape
            in_channels = conv_params[1]
            out_channels = conv_params[0]

            conv2d_layer_name = f'{layer_name}_{in_channels}_{out_channels}'  
            conv2d_dict[conv2d_layer_name].append(module)
            acsconv_dict[conv2d_layer_name].append(
                ACSConv(in_channels, out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups)
            )
            acsconv_dict[conv2d_layer_name][-1].load_state_dict(module.state_dict(), strict=False)

    return conv2d_dict, acsconv_dict


def create_new_acsconv_from_next_layer_acsconv(acsconv_dict, input_channels, output_channels, acsconv_index, stride, i):
    next_layer_acsconv_key = natsorted(acsconv_dict.keys())[i]
    next_layer_acsconv = acsconv_dict[next_layer_acsconv_key][acsconv_index]
    new_acsconv = ACSConv(
                        input_channels, output_channels, next_layer_acsconv.kernel_size,\
                        stride, next_layer_acsconv.padding,\
                        next_layer_acsconv.dilation, next_layer_acsconv.groups)
    new_acsconv.weight = nn.Parameter(
                        next_layer_acsconv.weight[:output_channels, :input_channels, :, :].clone()
                    )
    
    return next_layer_acsconv,new_acsconv
 

def replace_conv_with_acsconv(conv_dropout_norm_relu, acsconv, input_channels, output_channels, stride, acsconv_index):
    setattr(conv_dropout_norm_relu, "conv", acsconv)
    setattr(conv_dropout_norm_relu.all_modules, "0", acsconv)
    print("Successfully replace conv3d with pretrained acsconv {}->{} stride {} with index {}"\
                        .format(input_channels, output_channels, stride, acsconv_index))
    print("---------------------------------")


def replace_nnunet_encoder_conv3d_with_acs_pretrained(encoder, acsconv_dict):
    """
    Replace all 3D convolution layers in a given encoder with the corresponding pretrained ACSConv layers.

    Args:
        encoder (nn.Module): the encoder to modify
        acsconv_dict (dict): a dictionary mapping from ACSConv key strings to lists of ACSConv layers

    Returns:
        nn.Module: the modified encoder
    """
    stacked_conv_blocks = [k for k, m in encoder.named_modules(remove_duplicate=False) \
                           if isinstance(m, StackedConvBlocks) and not "decoder" in k]
    for parent in stacked_conv_blocks:
        conv_blocks = encoder.get_submodule(parent).convs
        for name, conv_dropout_norm_relu in conv_blocks.named_children():
            module = conv_blocks.get_submodule(name).conv
            input_channels, output_channels = module.weight.shape[1], module.weight.shape[0]
            desired_pretrained_acsconv_key = f"{input_channels}_{output_channels}"
            if input_channels == output_channels:
                idx = -1
            else:
                idx = 0
            for k, v in acsconv_dict.items():
                if k.endswith(desired_pretrained_acsconv_key) and v[idx].stride == module.stride:
                    replace_conv_with_acsconv(conv_dropout_norm_relu, v[idx], input_channels, output_channels, v[idx].stride[0], idx)
    return encoder


def replace_nnunet_encoder_conv3d_with_acs_pretrained_all(encoder, acsconv_dict):
    """
    Replace all 3D convolution layers in a given encoder with the corresponding pretrained ACSConv layers.

    Args:
        encoder (nn.Module): the encoder to modify
        acsconv_dict (dict): a dictionary mapping from ACSConv key strings to lists of ACSConv layers

    Returns:
        nn.Module: the modified encoder
    """
    stacked_conv_blocks = [k for k, m in encoder.named_modules(remove_duplicate=False) \
                           if isinstance(m, StackedConvBlocks) and not "decoder" in k]
    less_than_64_acsconv_index = 0
    same_320_acsconv_index = 0
    
    for parent in stacked_conv_blocks:
        conv_blocks = encoder.get_submodule(parent).convs
        
        for ix, (name, conv_dropout_norm_relu) in enumerate(conv_blocks.named_children()):
            module = conv_blocks.get_submodule(name).conv
            input_channels, output_channels = module.weight.shape[1], module.weight.shape[0]
            desired_pretrained_acsconv_key = f"{input_channels}_{output_channels}"
                      
            if input_channels == output_channels:
                if input_channels == 320:
                    acsconv_index = same_320_acsconv_index
                    same_320_acsconv_index += 1
                    stride = 1 if ix % 2 == 1 else 2
                else:
                    acsconv_index = -1
                    stride = 1
            else:
                acsconv_index = 0
                stride = 2 if output_channels > 32 else 1
                
            if output_channels <= 64:
                acsconv_index = less_than_64_acsconv_index
                less_than_64_acsconv_index += 1
                
            for i, (k, v) in enumerate(natsorted(acsconv_dict.items())):
                pretrained_input_channels, pretrained_output_channels = [int(x) for x in k.split('_')[-2:]]

                if k.endswith(desired_pretrained_acsconv_key) and v[acsconv_index].stride == stride:
                    replace_conv_with_acsconv(conv_dropout_norm_relu, v[acsconv_index], input_channels, output_channels, stride, acsconv_index)
                    break
                elif pretrained_input_channels >= input_channels and pretrained_output_channels >= output_channels:
                    print("The desired pretrained acsconv layer is {}->{} stride {}".format(input_channels, output_channels, module.stride[0]))
                    print("Unfortunatly, we cannot find a pretrained acsconv layer for this conv3d layer")
                    print("Trying to get the next larger pretrained acsconv layer and somehow init it")
                        
                    next_layer_acsconv, new_acsconv = create_new_acsconv_from_next_layer_acsconv(acsconv_dict, input_channels, output_channels, acsconv_index, stride, i)
                    replace_conv_with_acsconv(conv_dropout_norm_relu, new_acsconv, next_layer_acsconv.weight.shape[1], next_layer_acsconv.weight.shape[0], stride, acsconv_index)
                    break
    del acsconv_dict
    return encoder
