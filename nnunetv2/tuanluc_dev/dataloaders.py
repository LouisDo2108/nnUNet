import os
import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import Grayscale

from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.tuanluc_dev.utils import *
import monai.transforms as mt
import cv2

    
def get_BRATSDataset_dataloader(root_dir, batch_size, num_workers):
    
    train_transform = mt.Compose(
        [
            mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
        ]
    )
    val_transform = mt.Compose(
        [
            mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
        ]
    )
    
    train_dataset = BRATSDataset(root_dir, train=True, train_transform=train_transform, fold=0)
    val_dataset = BRATSDataset(root_dir, train=False, val_transform=val_transform, fold=0)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    sampler = StratifiedBatchSampler(train_dataset.labels, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


def slice_brats_utils(npy, filename, resize_transform, save_dir=None, save=False):
    npy = resize_transform(npy).numpy()
    map_modalities_dict = {0: "flair", 1: "t1", 2: "t1ce", 3: "t2"} # who knows the order?
    map_plans_dict = {0: "axial", 1: "coronal", 2: "sagittal"}
    
    for modalities_ix, modalities in enumerate(npy):
        for k in range(7, 128-8, 8):
            slices = get_slices_from_each_plan(modalities, k)
            if save:
                for plans_ix, slice in enumerate(slices):
                    if isinstance(slice, np.ndarray):
                        save_name = os.path.join(save_dir, f"{filename}_{map_modalities_dict[modalities_ix]}_{map_plans_dict[plans_ix]}_{k}.jpg")
                        cv2.imwrite(save_name, slice)
                        print(f"Saved {save_name}")
                    else:
                        print("found None slice")
                    

def normalize(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img


def get_slices_from_each_plan(modalities, k):
    slice_a = modalities[k, :, :]
    slice_c = modalities[:, k, :]
    slice_s = modalities[:, :, k]
    slices = [slice_a, slice_c, slice_s]
    
    for i in range(len(slices)):
        slices[i] = normalize(slices[i])
        slices[i] *= -255.0
        slices[i] = slices[i].astype(np.uint8)

    return slices


@timing
def slice_brats(
    brats_dir="/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018/nnUNetPlans_3d_fullres", 
    save_dir="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices",
    save=False):
    """
    Loop through all the npy files in the brats_dir and save the slices in another directory
    """
    brats_dir = Path(brats_dir)
    dataset_json = "/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset032_BraTS2018/splits_final.json"
    with open(dataset_json, 'r') as f:
        js = json.load(f)
    js = js[0]
    
    resize_transform = mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
    
    for k, v in js.items():
        _save_dir = Path(save_dir) / k
        _save_dir.mkdir(parents=True, exist_ok=True)
        for f in v:
            filename = brats_dir / f"{f}.npy" 
            print(filename, Path.is_file(filename))
            npy = np.load(filename)
            slice_brats_utils(npy, f, resize_transform, save_dir=_save_dir, save=save)


class BRATSDataset(Dataset):
    def __init__(self, root_dir, train, train_transform=None, val_transform=None, fold=0):
        self.root_dir = root_dir
        self.preprocessed_data_path = "/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018/nnUNetPlans_3d_fullres"
        
        self.label_dict = {}
        self.paths = []
        self.labels = []
        
        self.train = train
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        with open("/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018/splits_final.json", "r") as f:
            self.dataset_json = json.load(f)[0]
            
        for label in Path(root_dir).iterdir():
            if label.stem not in ['LGG', 'HGG']:
                continue
            for subject in label.iterdir():
                self.label_dict[subject.stem] = 0 if label.stem == 'LGG' else 1

        
        
        if self.train:
            for filename in self.dataset_json["train"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
            print("Train LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        else:
            for filename in self.dataset_json["val"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
            print("Val LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        data = np.load(path)
        if self.train and self.train_transform:
            data = self.train_transform(data)
        if not self.train and self.val_transform:
            data = self.val_transform(data)

        return data.as_tensor(), torch.tensor(float(label))


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        elif isinstance(y, list):
            y = np.array(y)
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        # return len(self.y)
        return self.n_batches


class ImageNetBRATSDataset(Dataset):
    def __init__(self, imagenet_json_path, brats_dir, train=True, train_transform=None, val_transform=None):
        with open(imagenet_json_path, 'r') as f:
            self.imagenet_json = json.load(f)['train' if train else 'val']
        self.brats_dir = Path(brats_dir) / 'train' if train else Path(brats_dir) / 'val'
        self.train = train
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.paths = self.imagenet_json + [x.resolve() for x in self.brats_dir.iterdir()]
        self.labels = np.concatenate([np.zeros(len(self.imagenet_json)), np.ones(len(self.paths) - len(self.imagenet_json))])
        self.grayscale = Grayscale(num_output_channels=3)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path, label = self.paths[idx], self.labels[idx]
        img = Image.open(img_path, mode='r').convert('RGB')
        if self.train_transform:
            img = self.train_transform(img)
        if self.val_transform:
            img = self.val_transform(img)
        img = self.grayscale(img)
        return img, label


def get_ImageNetBRATSDataset_dataloader(batch_size, num_workers):
    
    _, model_transform = get_model_and_transform("resnet18")
    train_transform = model_transform
    val_transform = model_transform
    
    brats_folder = "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices"
    imagenet_json_path = "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/imagenet_fold_0.json"
    
    train_dataset = ImageNetBRATSDataset(imagenet_json_path=imagenet_json_path, brats_dir=brats_folder, train=True, train_transform=train_transform)
    val_dataset = ImageNetBRATSDataset(imagenet_json_path=imagenet_json_path, brats_dir=brats_folder, train=False, val_transform=val_transform)
    
    # return train_dataset, val_dataset
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sampler = StratifiedBatchSampler(train_dataset.labels, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


if __name__ == '__main__':
    pass
    # set_seed(42)
    # train_loader, val_loader = get_BRATSDataset_dataloader(
    #     root_dir='/home/dtpthao/workspace/brats_projects/datasets/BraTS_2018/train',
    #     batch_size=4, num_workers=1
    # )
    # # train_loader, val_loader = get_ImageNetBRATSDataset_dataloader(
    # #     batch_size=4, num_workers=1
    # # )
    
    # for data, label in train_loader:
    #     print(data.shape, label.shape)
    #     print(label)
    #     # cv2.imwrite("/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/1.jpg", data.numpy()[0].transpose(1, 2, 0).astype(np.uint8))
    #     break
    
    # for data, label in val_loader:
    #     print(data.shape, label.shape)
    #     # cv2.imwrite("/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/2.jpg", data.numpy()[0].transpose(1, 2, 0).astype(np.uint8))
    #     break 
    # # slice_brats(save=True)
