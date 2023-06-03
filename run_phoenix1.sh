#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=1 # Change according to GPU availability
export nnUNet_n_proc_DA=9 # Change according to CPU availability, default is 12
nnUNet_raw="/tmp/htluc/nnunet/nnUNet_raw/"; export nnUNet_raw
nnUNet_preprocessed="/tmp/htluc/nnunet/nnUNet_preprocessed"; export nnUNet_preprocessed
nnUNet_results="/tmp/htluc/nnunet/nnUNet_results"; export nnUNet_results

eval "$(conda shell.bash hook)"
conda activate brats

cd /home/dtpthao/workspace/brats_projects/segmentations/nnUnet/nnunetv2


# name="acs_resnet18_encoder_all_bnda"
# name="acs_resnet18_encoder_bn"
name="cbam_baseline_everystage3"
# nnUNetTrainer_50epochs_tuanluc
# nnUNetTrainerDA_50epochs_tuanluc
# nnUNetTrainerBN_50epochs_tuanluc
# nnUNetTrainerBNDA_50epochs_tuanluc
# trainer="nnUNetTrainerBN_50epochs_tuanluc"
trainer="nnUNetTrainerCBAM_50epochs"
config_path="/home/dtpthao/workspace/brats_projects/segmentations/nnUnet/nnunetv2/configs/base.yaml"


# 0. Dataset conversion
# python dataset_conversion/Dataset032_BraTS2018.py

# 1. Plan + Preprocess
# python experiment_planning/plan_and_preprocess_entrypoints.py \
# -d 032 \
# -c "3d_fullres" \
# -np 4 \
# --verify_dataset_integrity \
# --clean \
# --verbose

# 2. Train + Val fold 0
python run/run_training.py 032 $name 0 -num_gpus 1 \
-tr $trainer \
-custom_cfg_path $config_path
# --c # Continue training

# (Optional) 2.1  find best config (Only viable after training all 5 folds)
# python evaluation/find_best_configuration.py 032 -c 3d_fullres_bs4_batch_dice -f 0 --disable_ensembling

### Test set
# 3. Test (nnUnet format)
# The -o (output folder) should locate in /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/{something}

train_test="test"
image_folder=$([ "$train_test" == "train" ] && echo "imagesTr" || echo "imagesTs")
python /home/dtpthao/workspace/nnUNet/nnunetv2/inference/predict_from_raw_data.py \
-i /tmp/htluc/nnunet/nnUNet_raw/Dataset032_BraTS2018/$image_folder \
-o /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/$name/fold_0/$train_test \
-d 032 \
-tr $trainer \
-c $name \
-f 0 \
-custom_cfg_path $config_path

# 4. Convert back to BraTS2018 format
python dataset_conversion/Dataset032_BraTS2018.py \
--exp-name $name \
--train $train_test

### Train set
# 3. Test (nnUnet format)
# The -o (output folder) should locate in /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/{something}
train_test="train"
image_folder=$([ "$train_test" == "train" ] && echo "imagesTr" || echo "imagesTs")
python /home/dtpthao/workspace/nnUNet/nnunetv2/inference/predict_from_raw_data.py \
-i /tmp/htluc/nnunet/nnUNet_raw/Dataset032_BraTS2018/$image_folder \
-o /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/$name/fold_0/$train_test \
-d 032 \
-tr $trainer \
-c $name \
-f 0 \
-custom_cfg_path $config_path

# 4. Convert back to BraTS2018 format
python dataset_conversion/Dataset032_BraTS2018.py \
--exp-name $name \
--train $train_test