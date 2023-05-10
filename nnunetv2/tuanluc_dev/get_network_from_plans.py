from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.tuanluc_dev.network_initialization import (
    init_weights_from_pretrained_proxy_task_encoder, 
    replace_nnunet_conv3d_with_acsconv_random,
    load_resnet18_custom_encoder,
    get_acs_pretrained_weights,
    replace_nnunet_encoder_conv3d_with_acs_pretrained,
    replace_nnunet_encoder_conv3d_with_acs_pretrained_all,
    InitWeights_He
)
from pprint import pprint
from pathlib import Path
from nnunetv2.tuanluc_dev.utils import *


def customize_network(model, custom_network_config_path):
    
    custom_network_config = read_custom_network_config(custom_network_config_path)
    
    if custom_network_config["proxy_encoder_class"] and custom_network_config["proxy_encoder_pretrained"]:
        """
        In case we want to replace the nnUNet encoder with a pretrained encoder from a proxy task
        Both proxy_encoder_class and proxy_encoder_pretrained must be specified
        """
        print("---------- REPLACE NNUNET ENCODER WITH CUSTOM PRETRAINED PROXY TASK ENCODER ----------")        
        try:
            # Simply pass in the config path
            model, pretrained_path = init_weights_from_pretrained_proxy_task_encoder(
                nnunet_model=model, 
                custom_network_config_path=custom_network_config_path
            )
            print("Successfully init weights from pretrained proxy task encoder from \n", pretrained_path)
        except Exception as e:
            print("Failed to init weights from pretrained proxy task encoder")
            print("Error: ", e)
            raise e
    elif custom_network_config["acsconv"]:
        """
        If no pretrained proxy task encoder is specified or any other reason
        Assume you want to replace conv3d with ACSConv
        Specify pretrained model using "acs_pretrained" key, random init if None
        Layer that are not loaded with pretrained weights will be init with nnUNet init (default),
        else will be init with ACSConv default init
        """
        print("---------- REPLACE CONV3D WITH ACSConv ----------")
        # acs_pretrained can be either timm pretrained or custom pretrained encoder
        # First, we will load the acs_conv pretrained weights as a dict
        # Then, we will replace the conv3d with acsconv
        
        # ------------------ Load ACSConv pretrained weights as a dict ------------------
        acsconv_dict = load_acsconv_dict(custom_network_config)
        
        # ------------------ Replace Conv3D with ACSConv with pretrained weight/random weight ------------------
        replace_conv3d_and_load_weight_from_acsconv(model, custom_network_config, acsconv_dict)
        
        # Always init with nnUNet init for non-pretrained layers after replacing conv3d with ACSConv
        # Otherwise, these layers are already init with ACSConv default init
        if custom_network_config["nnUNet_init"]:
            model.apply(InitWeights_He(1e-2))
            print("Successfully init with nnUNet init")
        else:
            print("Successfully init with ACSConv default init")
    # elif custom_network_config["conv_pretrained"] is not None:
    #     """
    #     Normal Conv3D nnUNet
    #     Can load custom pretrained encoder using "conv_pretrained" key
    #     Currently only support HGG/LGG classification pretrained encoder
    #     """
    #     print("---------- CONV3D NNUNET WITH PRETRAINED ENCODER ----------")
    #     # proxy_encoder_class: null # (Available: HGGLGGClassifier, ImageNetBratsClassifier)
    #     # proxy_encoder_pretrained: null
        
    #     # Init weights from pretrained encoder (Proxy task)
    #     conv_pretrained = custom_network_config["conv_pretrained"]
    #     try:
    #         model = init_weights_from_pretrained(
    #             nnunet_model=model, 
    #             pretrained_model_path=conv_pretrained
    #         )
    #         print("Successfully init weights from pretrained encoder from \n", conv_pretrained)
    #         print("Conv3D nnUNet is already init with nnUNet init") 
    #     except Exception as e:
    #         print("Failed to init weights from pretrained encoder")
    #         print("Error: ", e)
    #         raise e
    else:
        print("---------- DEFAULT CONV3D NNUNET ----------")
        print("Do nothing")
        
    print("--------------------------------------------------\n")


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


def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True,
                           custom_network_config_path: str = None,):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)

    if custom_network_config_path is not None:
        customize_network(model, custom_network_config_path)
        
    return model


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
