import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import nibabel as nib
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold

import random
import numpy as np

from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D

import monai.transforms as mt

# patch_size = [
#     128,
#     128,
#     128
# ]
# dim = len(patch_size)

# rotation_for_DA = {
#     'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
#     'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
#     'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
# }
# mirror_axes = (0, 1, 2)

# initial_patch_size = get_patch_size(patch_size[-dim:],
#                                             *rotation_for_DA.values(),
#                                             (0.85, 1.25))
# need_to_pad = (np.array(initial_patch_size) - np.array(patch_size)).astype(int)
# oversample_foreground_percent = 0.33


def set_seed(seed):
    # Set the seed for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            # print("Train transform")
            data = self.train_transform(data)
        if not self.train and self.val_transform:
            # print("Val transform")
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

        
def get_dataloader(root_dir, batch_size, num_workers):
    
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


if __name__ == '__main__':
    set_seed(42)
    train_loader, val_loader = get_dataloader(
        root_dir='/home/dtpthao/workspace/brats_projects/datasets/BraTS_2018/train',
        batch_size=4, num_workers=1)
    print(len(train_loader))
    print(len(val_loader))
    # for data, target in train_loader:
    #     print(target)
    #     break
    
    # for data, label in val_loader:
    #     print(data.shape, label.shape)
    #     break