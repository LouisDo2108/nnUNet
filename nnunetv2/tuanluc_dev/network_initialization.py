# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from nnunetv2.tuanluc_dev.encoder import HGGLGGClassifier
# from nnunetv2.tuanluc_dev.get_network_from_plans import get_encoder

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

if __name__ == "__main__":
    pass
    # nnunet_trainer = get_encoder()
    # nnunet_trainer = init_weights_from_pretrained(
    #     nnunet_trainer, 
    #     "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_95.pt"
    # )
    # print(nnunet_trainer.network.encoder)