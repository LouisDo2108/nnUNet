import multiprocessing
import shutil
from multiprocessing import Pool

import os
import argparse
from copy import deepcopy
from natsort import natsorted
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.tuanluc_dev.utils import zip_folder, post_processing
import json

json_file = "/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset082_BraTS2020/splits_final.json"

with open(json_file) as f:
    val = json.load(f)[0]["val"]

def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == '__main__':
    
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='The name of experiment')
    parser.add_argument('--train_test', type=str, help='Train or test')
    parser.add_argument('--root-dir', type=str, default="/home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/", help='Default folder to output result')
    parser.add_argument('--post', action='store_true', help='Enable post-processing')
    args = parser.parse_args()
    #    # 1. Dataset conversion (Train + Test) 
    #     brats_data_dir = '/home/dtpthao/workspace/nnUNet/data/BraTS_2020'
    #     brats_data_dir_train = '/home/dtpthao/workspace/nnUNet/data/BraTS_2020/MICCAI_BraTS2020_TrainingData'
    #     brats_data_dir_test = '/home/dtpthao/workspace/nnUNet/data/BraTS_2020/MICCAI_BraTS2020_ValidationData'

    #     task_id = 82
    #     task_name = "BraTS2020"

    #     foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    #     # setting up nnU-Net folders
    #     out_base = join(nnUNet_raw, foldername)
    #     imagestr = join(out_base, "imagesTr")
    #     imagests = join(out_base, "imagesTs")
    #     labelstr = join(out_base, "labelsTr")
        
    #     maybe_mkdir_p(imagestr)
    #     maybe_mkdir_p(imagests)
    #     maybe_mkdir_p(labelstr)

    #     case_ids_train = subdirs(brats_data_dir_train, prefix='BraTS', join=False)
    #     case_ids_test = subdirs(brats_data_dir_test, prefix='BraTS', join=False)

    #     for c in case_ids_train:
    #         shutil.copy(join(brats_data_dir_train, c, c + "_t1.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
    #         shutil.copy(join(brats_data_dir_train, c, c + "_t1ce.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
    #         shutil.copy(join(brats_data_dir_train, c, c + "_t2.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
    #         shutil.copy(join(brats_data_dir_train, c, c + "_flair.nii.gz"), join(imagestr, c + '_0003.nii.gz'))

    #         copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir_train, c, c + "_seg.nii.gz"),
    #                                                              join(labelstr, c + '.nii.gz'))
            
    #     for c in case_ids_test:
    #         shutil.copy(join(brats_data_dir_test, c, c + "_t1.nii.gz"), join(imagests, c + '_0000.nii.gz'))
    #         shutil.copy(join(brats_data_dir_test, c, c + "_t1ce.nii.gz"), join(imagests, c + '_0001.nii.gz'))
    #         shutil.copy(join(brats_data_dir_test, c, c + "_t2.nii.gz"), join(imagests, c + '_0002.nii.gz'))
    #         shutil.copy(join(brats_data_dir_test, c, c + "_flair.nii.gz"), join(imagests, c + '_0003.nii.gz'))

    #     generate_dataset_json(out_base,
    #                           channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
    #                           labels={
    #                               'background': 0,
    #                               'whole tumor': (1, 2, 3),
    #                               'tumor core': (2, 3),
    #                               'enhancing tumor': (3, )
    #                           },
    #                           num_training_cases=len(case_ids_train),
    #                           file_ending='.nii.gz',
    #                           regions_class_order=(1, 2, 3),
    #                           license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
    #                           reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
    #                           dataset_release='1.0')
    root_dir = "/home/dtpthao/workspace/nnUNet/env/results/Dataset082_BraTS2020/"
    exp_name = args.exp_name
    train_test = args.train_test
    print("-"*10, train_test, "-"*10)
    
    test_input_folder = os.path.join(root_dir, f"{exp_name}/fold_0/{train_test}")
    test_output_folder = os.path.join(root_dir, f"{exp_name}/fold_0/{train_test}_brats_format")
    
    print("Converting...")
    convert_folder_with_preds_back_to_BraTS_labeling_convention(
        test_input_folder,
        test_output_folder
    )
    
    zip_folder(test_output_folder, os.path.join(root_dir, f"{exp_name}/fold_0/{train_test}_results"))
    
    if args.post:
        print("Post processing...")
        post_input_folder = test_output_folder
        post_output_folder = os.path.join(root_dir, f"{exp_name}/fold_0/{train_test}_post_brats_format")
        Path(post_output_folder).mkdir(parents=True, exist_ok=True)
        
        for filename in natsorted(os.listdir(post_input_folder)):
            if filename.endswith(".nii.gz"):
                post_processing(filename, post_input_folder, post_output_folder, num_voxels=1000)
        zip_folder(post_output_folder, os.path.join(root_dir, f"{exp_name}/fold_0/{train_test}_results_post"))
    
    print("Completed converting labels back to BraTS2020 convention")
    print("Input folder: ", test_input_folder)
    print("Output folder: ", test_output_folder)
    if args.post:
        print("Post output folder: ", post_output_folder)