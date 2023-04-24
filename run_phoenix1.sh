#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=1 # Change according to GPU availability
export nnUNet_n_proc_DA=8 # Change according to CPU availability, default is 12
nnUNet_raw="/tmp/htluc/nnunet/nnUNet_raw/"; export nnUNet_raw
nnUNet_preprocessed="/tmp/htluc/nnunet/nnUNet_preprocessed"; export nnUNet_preprocessed
nnUNet_results="/tmp/htluc/nnunet/nnUNet_results"; export nnUNet_results

eval "$(conda shell.bash hook)"
conda activate nnunet

cd /home/dtpthao/workspace/nnUNet/nnunetv2

# 0. Dataset conversion
# python dataset_conversion/Dataset032_BraTS2018.py

# 1. Plan + Preprocess
# python experiment_planning/plan_and_preprocess_entrypoints.py \
# -d 032 \
# -c 3d_fullres \
# -np 4 \
# --verify_dataset_integrity \
# --clean \
# --verbose

# 2. Train + Val fold 0
python run/run_training.py 032 bs4_acs_resnet18_encoder_all 0 -num_gpus 1

# (Optional) 2.1  find best config (Only viable after training all 5 folds)
# python evaluation/find_best_configuration.py 032 -c 3d_fullres_bs4_batch_dice -f 0 --disable_ensembling

# 3. Test (nnUnet format)
# The -o (output folder) should locate in /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/{something}
# imagesTs imagesTr
# python /home/dtpthao/workspace/nnUNet/nnunetv2/inference/predict_from_raw_data.py \
# -i /tmp/htluc/nnunet/nnUNet_raw/Dataset032_BraTS2018/imagesTr \
# -o /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/nnUNetTrainer__nnUNetPlans__bs4_acs_resnet18_encoder/fold_0/train \
# -d 032 \
# -c bs4_acs_resnet18_encoder \
# -f 0 \
# --verbose

# 4. Convert back to BraTS2018 format
# python dataset_conversion/Dataset032_BraTS2018.py