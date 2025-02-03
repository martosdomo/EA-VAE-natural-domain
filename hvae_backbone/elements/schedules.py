import warnings

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from .. import params


def get_beta_schedule():
    """
    Get beta schedule for VAE
    Used to weight the KL loss of each group
    :return: nn.Module or uniform tensor function
    """
    if "vae_beta_max" in params.loss_params.keys():
        beta_max = params.loss_params.vae_beta_max
    else:
        beta_max = 1.
    return LogisticBetaSchedule(
        activation_step=params.loss_params.vae_beta_activation_steps,
        growth_rate=params.loss_params.vae_beta_growth_rate) \
        if params.loss_params.variation_schedule == 'Logistic' \
        else LinearBetaSchedule(
        anneal_start=params.loss_params.vae_beta_anneal_start,
        anneal_steps=params.loss_params.vae_beta_anneal_steps,
        contrast_beta_start=params.loss_params.contrast_beta_start,
        beta_min=params.loss_params.vae_beta_min,
        beta_max=beta_max) \
        if params.loss_params.variation_schedule == 'Linear' \
        else lambda x: torch.as_tensor(1.)


def get_schedule(optimizer, decay_scheme, warmup_steps, decay_steps, decay_rate, decay_start,
                 min_lr, last_epoch, checkpoint):
    """
    Get learning rate schedule

    :param optimizer: torch.optim.Optimizer, the optimizer to schedule
    :param decay_scheme: str, the decay scheme to use
    :param warmup_steps: int, the number of warmup steps
    :param decay_steps: int, the number of decay steps
    :param decay_rate: float, the decay rate
    :param decay_start: int, the number of steps before starting decay
    :param min_lr: float, the minimum learning rate
    :param last_epoch: int, the last epoch
    :param checkpoint: Checkpoint, the checkpoint to load the scheduler from

    :return: torch.optim.lr_scheduler.LRScheduler, the scheduler
    """
    if decay_scheme == 'noam':
        schedule = NoamSchedule(optimizer=optimizer, warmup_steps=warmup_steps, last_epoch=last_epoch)

    elif decay_scheme == 'exponential':
        schedule = NarrowExponentialDecay(optimizer=optimizer,
                                          decay_steps=decay_steps,
                                          decay_rate=decay_rate,
                                          decay_start=decay_start,
                                          minimum_learning_rate=min_lr,
                                          last_epoch=last_epoch)

    elif decay_scheme == 'cosine':
        schedule = NarrowCosineDecay(optimizer=optimizer,
                                     decay_steps=decay_steps,
                                     decay_start=decay_start,
                                     minimum_learning_rate=min_lr,
                                     last_epoch=last_epoch,
                                     warmup_steps=warmup_steps)

    elif decay_scheme == 'constant':
        schedule = ConstantLearningRate(optimizer=optimizer, last_epoch=last_epoch, warmup_steps=warmup_steps)

    else:
        raise NotImplementedError(f'{decay_scheme} is not implemented yet!')

    if checkpoint is not None:
        schedule.load_state_dict(checkpoint.scheduler_state_dict)
        print('Loaded Scheduler Checkpoint')

    return schedule


class LogisticBetaSchedule:
    """
    Logistic beta schedule for VAE
    from Efficient-VDVAE paper
    """
    def __init__(self, activation_step, growth_rate):
        self.beta_max = 1.
        self.activation_step = activation_step
        self.growth_rate = growth_rate

    def __call__(self, step):
        return self.beta_max / (1. + torch.exp(-self.growth_rate * (step - self.activation_step)))


class LinearBetaSchedule:
    """
    Linear beta schedule for VAE
    from Efficient-VDVAE paper
    """
    def __init__(self, anneal_start, anneal_steps, contrast_beta_start, beta_min, beta_max=1.):
        self.beta_max = beta_max
        self.anneal_start = anneal_start
        self.anneal_steps = anneal_steps
        self.contrast_beta_start = contrast_beta_start
        self.beta_min = beta_min

    def __call__(self, step):
        beta_1 = self.beta_min + (self.beta_max - self.beta_min) * torch.clamp(
             torch.tensor((step - self.anneal_start) / self.anneal_steps), min=0, max=1)
        if self.contrast_beta_start is None:
            return beta_1
        else:
            beta_2 = self.contrast_beta_start + (1 - self.contrast_beta_start) * torch.clamp(
                torch.tensor((step - self.anneal_start) / self.anneal_steps), min=0, max=1)
            return torch.tensor((beta_1, beta_2))
                           

class ConstantLearningRate(LRScheduler):
    """
    Constant learning rate scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        if warmup_steps != 0:
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = 1
        super(ConstantLearningRate, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return  [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps))
                for v in self.base_lrs]

    def _get_closed_form_lr(self):
        return [v * (torch.minimum(torch.tensor(1.), torch.tensor(self.last_epoch / self.warmup_steps)))
                for v in self.base_lrs]


class NarrowExponentialDecay(LRScheduler):
    """
    Narrow exponential learning rate decay scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, decay_steps, decay_rate, decay_start,
                 minimum_learning_rate, last_epoch=-1):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.decay_start = decay_start
        self.minimum_learning_rate = minimum_learning_rate

        super(NarrowExponentialDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ** ((self.last_epoch - self.decay_start) / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs

    def _get_closed_form_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ** ((self.last_epoch - self.decay_start) / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs


class NarrowCosineDecay(CosineAnnealingLR):
    """
    Narrow cosine learning rate decay scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, decay_steps, warmup_steps, decay_start=0, minimum_learning_rate=None, last_epoch=-1):
        self.decay_steps = decay_steps
        self.decay_start = decay_start
        self.minimum_learning_rate = minimum_learning_rate
        self.warmup_steps = warmup_steps

        assert self.warmup_steps <= self.decay_start

        super(NarrowCosineDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch, T_max=decay_steps,
                                                eta_min=self.minimum_learning_rate)

    def get_lr(self):
        if self.last_epoch < self.decay_start:
            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(NarrowCosineDecay, self).get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.decay_start:
            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(NarrowCosineDecay, self)._get_closed_form_lr()


class NoamSchedule(LRScheduler):
    """
    Noam learning rate scheduler
    from Efficient-VDVAE paper
    """
    def __init__(self, optimizer, warmup_steps=4000, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(NoamSchedule, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)
        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]
    
class EMAMomentumSchedule:
    
    """
    ipe := 1000 iterations per epoch
    num_epochs := 100 epochs
    
    example:
    0.996 + 0.004*step_n/(1000*100)
    """
    
    def __init__(self, ipe, num_epochs, ema0 = 0.996, ema1 = 1.0,):
        self.ema0 = ema0 
        self.ema1 = ema1
        self.ipe = ipe  # iterations per epoch
        self.num_epochs = num_epochs
        
    def __call__(self, step_n):
        return max(
            self.ema0 + step_n*(self.ema1-self.ema0)/(self.ipe*self.num_epochs),
            1.0
        )
