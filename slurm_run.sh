#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=phoenix3
#SBATCH --time=30-00:00:00
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G         # memory per cpu-core (4G is default)
eval "$(conda shell.bash hook)"
conda activate brats

cd /home/dtpthao/workspace/brats_projects/segmentations/nnUnet/nnunetv2
# python dataset_conversion/convert_MSD_dataset.py \
# -i /home/dtpthao/workspace/nnUNet/data/Task02_Heart

# python experiment_planning/plan_and_preprocess_entrypoints.py \
# -d 004 \
# --verify_dataset_integrity \

# python experiment_planning/plan_and_preprocess_entrypoints.py \
# -d 032 \
# -c 3d_fullres \
# -np 4 \
# --verify_dataset_integrity \
# --clean \
# --verbose

python run/run_training.py 032 "3d_fullres" 0 -num_gpus 1
