import torch
from torch import nn

from const import const
from learning.utils import compute_class_weights


def choose_criterion(key, params, class_weights):
    if key == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif key == 'WCELoss':
        # class_weights = compute_class_weights(train_loader, num_classes)
        # sum_weights = sum(class_weights)
        # class_weights = [w / sum_weights for w in class_weights]
        class_weights_tensor = torch.FloatTensor(class_weights).to(const._DEVICE)
        # print('Using a weighted CE loss for gait impairment score prediction.')
        # print("Class Weights:", class_weights)
        return nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif key == 'FocalLoss':
        return FocalLoss(params['alpha'], params['gamma'])
    elif key == 'WCELoss+smoothing':
        #TODO
        weights = class_weights #torch.tensor([88., 131., 180.])
        weights = weights / weights.sum()  # turn into percentage
        weights = 1.0 / weights  # inverse
        weights = weights / weights.sum()
        loss_weights = weights.to(const._DEVICE)
        print('Using a weighted *Smoothing CE loss* for gait impairment score prediction.')
        WeightedCrossEntropyLossWithLabelSmoothing(weight=loss_weights, smoothing=params['smoothing_scale'])
    else:
        raise ModuleNotFoundError("Criterion does not exist")

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss

# Define cross-entropy loss with label smoothing
class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(CrossEntropyLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
    def forward(self, inputs, targets):
        num_classes = inputs.size()[-1]
        log_preds = nn.functional.log_softmax(inputs, dim=-1)
        targets = torch.zeros_like(log_preds).scatter_(-1, targets.unsqueeze(-1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
        loss = nn.functional.kl_div(log_preds, targets, reduction='batchmean')
        return loss


class WeightedCrossEntropyLossWithLabelSmoothing(nn.Module):
  def __init__(self, weight, smoothing=0.1):
    super(WeightedCrossEntropyLossWithLabelSmoothing, self).__init__()
    self.smoothing = smoothing
    self.weight = weight

  def forward(self, inputs, targets):
    num_classes = inputs.size()[-1]
    log_preds = nn.functional.log_softmax(inputs, dim=-1)
    targets = torch.zeros_like(log_preds).scatter_(-1, targets.unsqueeze(-1), 1)
    targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
    loss = nn.functional.kl_div(log_preds, targets, reduction='none')
    loss = loss * self.weight.unsqueeze(0)
    loss = loss.sum(dim=-1).mean()
    return loss