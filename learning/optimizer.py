from torch import optim
from torch.optim.lr_scheduler import StepLR


def choose_scheduler(optimizer, params):
    scheduler_name = params.get('scheduler')
    if scheduler_name is None:
        print("[WARN] LR Scheduler is not used")
        return None

    if scheduler_name == "StepLR":
        scheduler = StepLR(optimizer, step_size=params['lr_step_size'], gamma=params['lr_decay'])
    else:
        raise ModuleNotFoundError("Scheduler is not defined")

    return scheduler


def choose_optimizer(model, params):
    optimizer_name = params['optimizer']
    try:
        backbone_params = set(model.module.backbone.parameters())
        head_params = set(model.module.head.parameters())
    except AttributeError:
        backbone_params = set(model.backbone.parameters())
        head_params = set(model.head.parameters())
        
    all_params = set(model.parameters())
    other_params = all_params - backbone_params - head_params
        
    param_groups = [
        {"params": filter(lambda p: p.requires_grad, backbone_params), "lr": params['lr_backbone'], 'weight_decay': params['weight_decay_backbone']},
        {"params": filter(lambda p: p.requires_grad, head_params), "lr": params['lr_head'], 'weight_decay': params['weight_decay']},
        {"params": filter(lambda p: p.requires_grad, other_params), "lr": params['lr_head'], 'weight_decay': params['weight_decay']}
    ]
            
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(param_groups)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(param_groups)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(param_groups, momentum=params.get('momentum', 0.9), weight_decay=params.get('weight_decay', 0.0))
    else:
        raise ModuleNotFoundError("Optimizer not found")

    return optimizer
