import os

import torch
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def upload_checkpoints_to_wandb(latest_epoch_path, best_epoch_path):
    artifact = wandb.Artifact(f'model', type='model')
    artifact.add_file(latest_epoch_path)
    artifact.add_file(best_epoch_path)
    wandb.log_artifact(artifact)


def save_checkpoint(checkpoint_root_path, epoch, lr, optimizer, model, fold, latest):
    checkpoint_path_fold = os.path.join(checkpoint_root_path, f"fold{fold}")
    if not os.path.exists(checkpoint_path_fold):
        os.makedirs(checkpoint_path_fold)
    checkpoint_path = os.path.join(checkpoint_path_fold,
                                   'latest_epoch.pth.tr' if latest else 'best_epoch.pth.tr')
    torch.save({
        'epoch': epoch,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }, checkpoint_path)


def assert_learning_params(params):
    """Makes sure the learning parameters is set as parameters (To avoid raising error during training)"""
    learning_params = ['batch_size', 'criterion', 'optimizer', 'lr_backbone', 'lr_head', 'weight_decay', 'epochs']
    for learning_param in learning_params:
        assert learning_param in params, f'"{learning_param}" is not set in params.'

def compute_class_weights(data_loader, params):
    class_counts = Counter()
    total_samples = 0
    num_classes = 0

    for _, targets, _, _, _ in data_loader:
        class_counts.update(targets.tolist())
        total_samples += len(targets)

    class_weights = []

    num_classes = params['num_classes']
    for i in range(num_classes):
        count = class_counts[i]
        weight = 0.0 if count == 0 else total_samples / (num_classes * count)
        class_weights.append(weight)
        
        total_weights = sum(class_weights)
        normalized_class_weights = [weight / total_weights for weight in class_weights]

    return normalized_class_weights

def log_cfm_to_wandb(confusion, fold, num_classes, kind='val'):
    fig, ax = plt.subplots(figsize=(10, 8)) 
    sns.heatmap(confusion, annot=True, ax=ax, cmap="Blues" if kind == 'val' else 'Greens', fmt='g', annot_kws={"size": 26})
    ax.set_xlabel('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    ax.set_title('Confusion Matrix', fontsize=30)
    ax.xaxis.set_ticklabels([f'class {i}' for i in range(num_classes)], fontsize=22)  # Modify class names as needed
    ax.yaxis.set_ticklabels([f'class {i}' for i in range(num_classes)], fontsize=22)
    wandb.log({f"cfm_{kind}_fold{fold}": wandb.Image(fig)})
    plt.close()
