# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from nnunetv2.tuanluc_dev.get_network_from_plans_dev import get_encoder
from torchsummary import summary
from pprint import pprint
from nnunetv2.tuanluc_dev.utils import *


if __name__ == '__main__':

    nnunet_trainer = get_encoder()
    print(nnunet_trainer.network.encoder)