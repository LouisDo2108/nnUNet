import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary

from nnunetv2.tuanluc_dev.dataloaders import get_dataloader
from nnunetv2.tuanluc_dev.encoder import HGGLGGClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, output_folder):
    # Plot the training and validation losses
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot as a PNG picture
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def train(model, train_loader, val_loader, 
          output_folder="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/checkpoints",
          num_epochs=100, learning_rate=0.01):
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, nesterov=True)
    scheduler = PolyLRScheduler(optimizer, learning_rate, num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
     # create output folder if not exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
     # create output folder if not exist
    Path(os.path.join(output_folder, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    train_losses = []
    val_losses = []
    
    pbar = tqdm(range(1, num_epochs+1))
    for epoch in pbar:
        
        pbar.set_description(f"Epoch {epoch}")
        model.train()
        train_loss = 0.0
        
        # Train
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validate on validation set every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.float().to(device)
                    output = model(data)
                    val_loss += F.binary_cross_entropy_with_logits(output.squeeze(1), target, reduction='sum').item()
                    pred = torch.sigmoid(output).round().squeeze(1)
                    predictions.extend(pred.tolist())
                    true_labels.extend(target.tolist())

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            log_metrics(output_folder, epoch, val_loss, predictions, true_labels, train=False)

            # save the model
            torch.save(model.state_dict(), os.path.join(output_folder, 'checkpoints', 'model_{:02d}.pt'.format(epoch)))
            log_loss(output_folder, epoch, val_loss, train=False)
            plot_loss(train_losses, val_losses, output_folder)
            
        scheduler.step(train_loss)
        log_loss(output_folder, epoch, train_loss, train=True)


def log_loss(output_folder, epoch, train_loss, train=False):
    train_val = 'Train' if train else 'Validation'
    with open(os.path.join(output_folder, '{}_loss.txt'.format(train_val)), 'a') as f:
        f.write('Epoch: {} \t{} Loss: {:.6f}\n'.format(epoch, train_val, train_loss))


def log_metrics(output_folder, epoch, loss, predictions, true_labels, train=False):
    train_val = 'Train' if train else 'Validation'
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)
            
    with open(os.path.join(output_folder, '{}_metrics.txt'.format(train_val)), 'a') as f:
        f.write('Epoch: {:<5}\n'.format(epoch))
        f.write('{:<10} Average Loss:  {:.4f}\n'.format(train_val, loss))
        f.write('{:<10} Accuracy:      {:.4f}\n'.format(train_val, accuracy))
        f.write('{:<10} Precision:     {:.4f}\n'.format(train_val, precision))
        f.write('{:<10} Recall:        {:.4f}\n'.format(train_val, recall))
        f.write('{:<10} F1 Score:      {:.4f}\n'.format(train_val, f1))
        f.write('{:<10} ROC AUC Score: {:.4f}\n'.format(train_val, roc_auc))


if __name__ == '__main__':
    
    train_loader, val_loader = get_dataloader(
        root_dir='/home/dtpthao/workspace/brats_projects/datasets/BraTS_2018/train',
        batch_size=4, num_workers=1)
    
    model = HGGLGGClassifier(4, 2)
    # summary(model, (4, 128, 128, 128))

    train(model, train_loader, val_loader, 
          output_folder="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg", 
          num_epochs=100, learning_rate=0.01)