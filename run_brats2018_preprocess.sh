#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=0 # Change according to GPU availability
export nnUNet_n_proc_DA=8 # Change according to CPU availability, default is 12

eval "$(conda shell.bash hook)"
conda activate nnunet
cd /home/dtpthao/workspace/nnUNet/

name="baseline"
trainer="nnUNetTrainer_50epochs_tuanluc"
config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/base.yaml"
fold=0

# 0. Dataset conversion
python dataset_conversion/Dataset032_BraTS2018.py

# 1. Plan + Preprocess
python experiment_planning/plan_and_preprocess_entrypoints.py \
-d 032 \
-c "3d_fullres" \
-np 8 \
--verify_dataset_integrity \
--clean \
--verbose