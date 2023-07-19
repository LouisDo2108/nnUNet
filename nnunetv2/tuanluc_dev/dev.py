# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from nnunetv2.tuanluc_dev.get_network_from_plans_dev import get_encoder
from torchsummary import summary
from pprint import pprint
from nnunetv2.tuanluc_dev.utils import *
import numpy
from pathlib import Path

if __name__ == '__main__':
    
    # npy_dir = "/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data/train/npy"
    # nifti_dir = "/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data/train/NIfTI"
    # print(len(list(Path(npy_dir).iterdir())))
    # print(len(list(Path(nifti_dir).iterdir())))
    
    root_dir = Path("/home/dtpthao/workspace/nnUNet/env/raw/Dataset032_BraTS2018_Nicegan")
    train_dir = root_dir / "imagesTr"
    test_dir = root_dir / "imagesTs"
    
    for file in test_dir.iterdir():
        filename = file.stem.split('.')[0]
        if file.is_file() and (filename.endswith("0001") or filename.endswith("0003")):
            file.unlink()
            print(f"Deleted {file}")
            # break