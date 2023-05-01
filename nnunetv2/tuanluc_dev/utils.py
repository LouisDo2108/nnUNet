import time
import random
import numpy as np
import torch


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.perf_counter()
        ret = f(*args, **kwargs)
        time2 = time.perf_counter()
        print('%s function took %0.4fs' % (f.__name__, (time2 - time1)))
        return ret
    return wrap


def set_seed(seed):
    # Set the seed for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def print_layers(model, target_layer):
    try:
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                print_layers(module, target_layer)
            
            if isinstance(module, target_layer):
                print(module)
                print('-----------------------')
    except Exception as e:
        print(e)

def create_imagenet_val_json():
    import os
    import json
    from sklearn.model_selection import KFold

    # Path to the ImageNet folder
    imagenet_folder = "/home/dtpthao/workspace/datasets/imagenet_val_2017"

    # Number of folds to use
    num_folds = 5

    # Get list of all image files in the folder
    image_files = [str(x) for x in Path(imagenet_folder).iterdir() if x.is_file() and x.suffix in [".JPEG", ".jpg"]]
    print(len(image_files))
    # # Split the image files into train and validation sets using k-fold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    train_indices, val_indices = list(kf.split(image_files))[0]
    train_files = [image_files[i] for i in train_indices]
    val_files = [image_files[i] for i in val_indices]

    # Create a dictionary containing the paths of train and validation image folders
    data = {
        "train": train_files,
        "val": val_files
    }

    # Save the dictionary as a JSON file
    with open("/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/imagenet_fold_0.json", "w") as f:
        json.dump(data, f)
        

def get_model_and_transform(model_name, pretrained=True):
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm import create_model
    
    model = create_model(model_name, pretrained=pretrained)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform