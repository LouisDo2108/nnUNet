# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from nnunetv2.tuanluc_dev.encoder import HGGLGGClassifier
from nnunetv2.tuanluc_dev.acsconv.operators import ACSConv
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
import timm
from collections import defaultdict
from natsort import natsorted
from nnunetv2.tuanluc_dev.encoder import ImageNetBratsClassifier

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


@torch.no_grad()
def init_weights_from_pretrained(nnunet_model, pretrained_model_path):
    pretrained_model = HGGLGGClassifier(4, 2, return_skips=True)
    loaded = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    pretrained_model.load_state_dict(loaded, strict=False)
    del loaded
    pretrained_encoder = pretrained_model.encoder
    nnunet_model.encoder = pretrained_encoder.to(torch.device('cuda'))
    return nnunet_model


def replace_nnunet_conv3d_with_acsconv_random(model, target_layer):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_nnunet_conv3d_with_acsconv_random(module, target_layer)
            
        if isinstance(module, target_layer):
            setattr(model, n, ACSConv(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups))


def load_resnet18_custom_encoder(pretrained_model_path):
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


if __name__ == "__main__":
    pass
    # nnunet_trainer = get_encoder()
    # nnunet_trainer = init_weights_from_pretrained(
    #     nnunet_trainer, 
    #     "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_95.pt"
    # )
    # print(nnunet_trainer.network.encoder)