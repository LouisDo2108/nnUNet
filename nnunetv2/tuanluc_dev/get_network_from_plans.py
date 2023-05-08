from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.tuanluc_dev.network_initialization import (
    init_weights_from_pretrained, 
    replace_nnunet_conv3d_with_acsconv_random,
    load_resnet18_custom_encoder,
    get_acs_pretrained_weights,
    replace_nnunet_encoder_conv3d_with_acs_pretrained,
    replace_nnunet_encoder_conv3d_with_acs_pretrained_all
)
from pprint import pprint
from pathlib import Path
from nnunetv2.run.run_training import run_training_entry
from nnunetv2.tuanluc_dev.utils import *


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
    
    # custom_network_config = {
    #     "acsconv": False,
    #     "conv_pretrained": None, # HGG/LGG pretrained
    #     "acs_pretrained": "resnet18",
    #     "replace_all": False, # Which mode to choose, all or not all
    #     "nnUNet_init": True,
    # }
    # "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/resnet18_brats_imagenet_encoder/checkpoints/model_10.pt"
    
    print("---------- Custom network config ----------")
    try:
        print("Config file name:", Path(custom_network_config_path).name)
        custom_network_config = load_yaml_config_file(custom_network_config_path)
        pprint(custom_network_config)
    except Exception as e:
        print(e)
    print("--------------------------------------------------\n")
    
    if custom_network_config["acsconv"]:
        """
        Replace conv3d with ACSConv
        Specify pretrained model using "acs_pretrained" key, random init if None
        Layer that are not loaded with pretrained weights will be init with nnUNet init (default),
        else will be init with ACSConv default init
        """
        print("---------- REPLACE CONV3D WITH ACSConv ----------")
        # acs_pretrained can be either timm pretrained or custom pretrained encoder
        # First, we will load the acs_conv pretrained weights as a dict
        # Then, we will replace the conv3d with acsconv
        
        # ------------------ Load ACSConv pretrained weights as a dict ------------------
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
        
        # ------------------ Replace Conv3D with ACSConv ------------------
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
        
        # Always init with nnUNet init for non-pretrained layers after replacing conv3d with ACSConv
        # Otherwise, these layers are already init with ACSConv default init
        if custom_network_config["nnUNet_init"]:
            model.apply(InitWeights_He(1e-2))
            print("Successfully init with nnUNet init")
        else:
            print("Successfully init with ACSConv default init")
    elif custom_network_config["conv_pretrained"] is not None:
        """
        Normal Conv3D nnUNet
        Can load custom pretrained encoder using "conv_pretrained" key
        Currently only support HGG/LGG classification pretrained encoder
        """
        print("---------- CONV3D NNUNET WITH PRETRAINED ENCODER ----------")
        
        # Init weights from pretrained encoder (Proxy task)
        conv_pretrained = custom_network_config["conv_pretrained"]
        try:
            #"/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_55.pt"
            model = init_weights_from_pretrained(
                nnunet_model=model, 
                pretrained_model_path=conv_pretrained
            )
            print("Successfully init weights from pretrained encoder from \n", conv_pretrained)
            print("Conv3D nnUNet is already init with nnUNet init") 
        except Exception as e:
            print("Failed to init weights from pretrained encoder")
            print("Error: ", e)
            raise e
    else:
        print("---------- DEFAULT NNUNET ----------")
        print("Do nothing")
        
    print("--------------------------------------------------\n")
    return model
    
    # ------------------ Custom network config ------------------
    # # Replace encoder conv with ACSConv pretrained
    
    # # Using custom pretrained resnet18 encoder
    # # pretrained_resnet18_path = "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/resnet18_brats_imagenet_encoder/checkpoints/model_10.pt"
    # # model_name = load_resnet18_custom_encoder(pretrained_resnet18_path)
    # # model_name = 'resnet18'
    # # _, acsconv_dict = get_acs_pretrained_weights(model_name=model_name)
    
    # # Replace conv with ACSConv
    # try:
    #     # replace_nnunet_conv3d_with_acsconv_random(model, nn.Conv3d)
    #     # print("Successfully replaced conv3d with ACSConv")
        
    #     model.apply(InitWeights_He(1e-2)) # nnUNet init for ACSConv, disable if using ACS default init
        
    #     # replace_nnunet_encoder_conv3d_with_acs_pretrained(model.encoder, acsconv_dict)
    #     # print("Successfully load pretrained weights for ACSConv")
        
    #     # replace_nnunet_encoder_conv3d_with_acs_pretrained_all(model.encoder, acsconv_dict)
    #     # print("Successfully load pretrained weights for all ACSConv")
    # except Exception as e:
    #     print(e)
        
    # # Init weights from pretrained encoder (Proxy task)
    # # model = init_weights_from_pretrained(
    # #     nnunet_model=model, 
    # #     pretrained_model_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_55.pt")
    
    # return model
