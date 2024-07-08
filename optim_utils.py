from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import AdamW


class WarmupLinearScheduleNonZero(_LRScheduler):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
        Linearly decreases learning rate linearly to min_lr over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, min_lr=1e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinearScheduleNonZero, self).__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_factor = float(step) / float(max(1, self.warmup_steps))
        else:
            lr_factor = max(0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

        return [base_lr * lr_factor if (base_lr * lr_factor) > self.min_lr else self.min_lr for base_lr in self.base_lrs]


def init_optim(model, config):
    encoder_params_with_weight_decay = []
    encoder_params_without_weight_decay = []
    decoder_params_with_weight_decay = []
    decoder_params_without_weight_decay = []
    other_params_with_weight_decay = []
    other_params_without_weight_decay = []

    exclude_from_weight_decay=['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # Our model shares (embedding) parameters between the encoder and decoder.
    # We want to include such parameters only in one parameter group.
    # So we keep track of the unique ids of each parameter. 
    params_ids = []

    for module_name, module in model.named_children():
        for param_name, param in module.named_parameters():
            if id(param) not in params_ids:
                params_ids.append(id(param))
            else:
                continue
            if param.requires_grad:
                if 'encoder' in param_name:
                    if any(ex in param_name for ex in exclude_from_weight_decay):
                        encoder_params_without_weight_decay.append(param)
                    else:
                        encoder_params_with_weight_decay.append(param)
                           
                elif 'decoder' in param_name:
                    if any(ex in param_name for ex in exclude_from_weight_decay):
                        decoder_params_without_weight_decay.append(param)
                    else:
                        decoder_params_with_weight_decay.append(param)
                else:
                    if any(ex in param_name for ex in exclude_from_weight_decay):
                        other_params_without_weight_decay.append(param)
                    else:
                        other_params_with_weight_decay.append(param)

    optimizer_grouped_parameters = [
        {
            'params': encoder_params_with_weight_decay,
            'weight_decay': 0.01,
            'lr': config['learning_rate_bart']
        },
        {
            'params': encoder_params_without_weight_decay,
            'weight_decay': 0.0,
            'lr': config['learning_rate_bart']
        },
        {
            'params': decoder_params_with_weight_decay,
            'weight_decay': 0.01,
            'lr': config['learning_rate_bart']
        },
        {
            'params': decoder_params_without_weight_decay,
            'weight_decay': 0.0,
            'lr': config['learning_rate_bart']
        },
        {
            'params': other_params_with_weight_decay,
            'weight_decay': 0.01,
            'lr': config['learning_rate_other']
        },
        {
            'params': other_params_without_weight_decay,
            'weight_decay': 0.0,
            'lr': config['learning_rate_other']
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate_bart'])

    scheduler = WarmupLinearScheduleNonZero(
        optimizer,
        warmup_steps=config['warmup_steps'],
        t_total=config['train_steps'],
        min_lr=config['min_lr']
    )

    return optimizer, scheduler
