#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=1 # Change according to GPU availability
export nnUNet_n_proc_DA=8 # Change according to CPU availability, default is 12

# nnUNet_raw="/tmp/htluc/nnunet/nnUNet_raw/"; export nnUNet_raw
# nnUNet_preprocessed="/tmp/htluc/nnunet/nnUNet_preprocessed"; export nnUNet_preprocessed
# nnUNet_results="/tmp/htluc/nnunet/nnUNet_results"; export nnUNet_results

eval "$(conda shell.bash hook)"
conda activate nnunet
cd /home/dtpthao/workspace/nnUNet/nnunetv2

name="baseline"
trainer="nnUNetTrainer_50epochs_tuanluc"
config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/base.yaml"
fold=4

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
# python run/run_training.py 032 $name $fold -num_gpus 1 \
# -tr $trainer \
# -custom_cfg_path $config_path \
# --c # Continue training

# Find best config (Only viable after training all 5 folds)
# python evaluation/find_best_configuration.py 032 -c $name -tr $trainer

# Run inference
# train_test="test"
# image_folder=$([ "$train_test" == "train" ] && echo "imagesTr" || echo "imagesTs")
# nnUNetv2_predict \
# -d Dataset032_BraTS2018 \
# -i /tmp/htluc/nnunet/nnUNet_raw/Dataset032_BraTS2018/$image_folder \
# -o /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/nnUNetTrainer_50epochs_tuanluc__nnUNetPlans__baseline/inference \
# -f  0 1 2 3 4 -tr nnUNetTrainer_50epochs_tuanluc -c baseline -p nnUNetPlans \
# -custom_cfg_path $config_path

# nnUNetv2_apply_postprocessing \
# -i /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/nnUNetTrainer_50epochs_tuanluc__nnUNetPlans__baseline/inference \
# -o /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/nnUNetTrainer_50epochs_tuanluc__nnUNetPlans__baseline/inference_post \
# -pp_pkl_file /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/nnUNetTrainer_50epochs_tuanluc__nnUNetPlans__baseline/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
# -np 8 \
# -plans_json /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/nnUNetTrainer_50epochs_tuanluc__nnUNetPlans__baseline/crossval_results_folds_0_1_2_3_4/plans.json

# python dataset_conversion/Dataset032_BraTS2018.py \
# --exp-name baseline \
# --train test \
# --fold $fold

### Test set
# 3. Test (nnUnet format)
# The -o (output folder) should locate in /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/{something}
# train_test="test"
# image_folder=$([ "$train_test" == "train" ] && echo "imagesTr" || echo "imagesTs")
# python /home/dtpthao/workspace/nnUNet/nnunetv2/inference/predict_from_raw_data.py \
# -i /tmp/htluc/nnunet/nnUNet_raw/Dataset032_BraTS2018/$image_folder \
# -o /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/$name/fold_$fold/$train_test \
# -d 032 \
# -tr $trainer \
# -c $name \
# -f $fold \
# -custom_cfg_path $config_path \
# --save_probabilities # Only if you want to do ensembling, very disk space consuming

# # # 4. Convert back to BraTS2018 format
# python dataset_conversion/Dataset032_BraTS2018.py \
# --exp-name $name \
# --train $train_test \
# --post \
# --fold $fold

# ### Train set
# # 3. Test (nnUnet format)
# # The -o (output folder) should locate in /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/{something}
# train_test="train"
# image_folder=$([ "$train_test" == "train" ] && echo "imagesTr" || echo "imagesTs")
# python /home/dtpthao/workspace/nnUNet/nnunetv2/inference/predict_from_raw_data.py \
# -i /tmp/htluc/nnunet/nnUNet_raw/Dataset032_BraTS2018/$image_folder \
# -o /home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/$name/fold_0/$train_test \
# -d 032 \
# -tr $trainer \
# -c $name \
# -f 0 \
# -custom_cfg_path $config_path

# # 4. Convert back to BraTS2018 format
# python dataset_conversion/Dataset032_BraTS2018.py \
# --exp-name $name \
# --train $train_test \
# --post