import math
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.fmn_config import *
from src.utils.projection import l0_projection, l1_projection, linf_projection, l2_projection
from src.utils.projection import l0_mid_points, l1_mid_points, l2_mid_points, linf_mid_points


class FMN:
    r"""
    FMN in the paper 'Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints'
    [https://arxiv.org/abs/2102.12827]

    Distance Measure : L0, L1, L2, Linf

    Args:
        model (nn.Module): The neural network model to be attacked.
        norm (float): The norm for the distance measure. Default: Linf (float('inf'))
        steps (int): The number of optimization steps. Default: 100
        alpha_init (float): The initial value of the optimization step size.
        alpha_final (float, optional): The final value of the optimization step size. Default is None.
        gamma_init (float): The initial value of the epsilon decay. Default: 0.05
        gamma_final (float): The final value of the epsilon decay. Default: 0.001.
        binary_search_steps (int): The number of binary search steps for the boundary search.
        starting_points (Tensor, optional): The starting points for optimization. Default is None.
        loss (Literal['LL', 'CE', 'DLR']): The type of loss function to be used. Default is 'LL'.
        optimizer (Literal['SGD', 'Adam', 'Adamax']): The optimizer to be used. Default is 'SGD'.
        scheduler (Literal['CALR', 'RLROP', None]): The learning rate scheduler to be used. Default is 'CALR'.
        optimizer_config (tuple, list, optional): The configuration for the optimizer. Default is None.
        scheduler_config (tuple, list, optional): The configuration for the scheduler. Default is None.
        targeted (bool): Whether the attack is targeted or not. Default is ``False``.
        verbose (bool): Whether to print log information or not. Default is ``False``.
        device (torch.device): The device on which to run the model. Default is torch.device('cpu').
    """

    def __init__(self,
                 model: nn.Module,
                 norm: float = torch.inf,
                 steps: int = 10,
                 alpha_init: float = 1.0,
                 alpha_final: Optional[float, None] = None,
                 gamma_init: float = 0.05,
                 gamma_final: float = 0.001,
                 binary_search_steps: int = 10,
                 starting_points: Optional[Tensor, None] = None,
                 loss: Literal['LL', 'CE', 'DLR'] = 'LL',
                 optimizer: Literal['SGD', 'Adam', 'Adamax'] = 'SGD',
                 scheduler: Literal['CALR', 'RLROP', None] = 'CALR',
                 optimizer_config: Optional[tuple, list, None] = None,
                 scheduler_config: Optional[tuple, list, None] = None,
                 targeted: bool = False,
                 verbose: bool = False,
                 device: torch.device = torch.device('cpu')
                 ):
        self.model = model
        self.norm = norm
        self.steps = steps
        self.alpha_init = alpha_init
        self.alpha_final = self.alpha_init / 100 if alpha_final is None else alpha_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.starting_points = starting_points
        self.binary_search_steps = binary_search_steps
        self.targeted = targeted
        self.loss = loss
        self.verbose = verbose
        self.device = device

        self.loss = LOSSES.get(loss, LL)
        self.optimizer = OPTIMIZERS.get(optimizer, SGD)
        self.scheduler = SCHEDULERS.get(scheduler, CosineAnnealingLR)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self._dual_projection_mid_points = {
            0: (None, l0_projection, l0_mid_points),
            1: (float('inf'), l1_projection, l1_mid_points),
            2: (2, l2_projection, l2_mid_points),
            float('inf'): (1, linf_projection, linf_mid_points),
        }

    def _boundary_search(self, images, labels):
        batch_size = len(images)
        _, _, mid_point = self._dual_projection_mid_points[self.norm]

        is_adv = self.model(self.starting_points).argmax(dim=1)
        if not is_adv.all():
            raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.ones(batch_size, device=self.device)
        for _ in range(self.binary_search_steps):
            epsilon = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=images, x1=self.starting_points, epsilon=epsilon)
            pred_labels = self.model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if self.targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, epsilon)
            upper_bound = torch.where(is_adv, epsilon, upper_bound)

        delta = mid_point(x0=images, x1=self.starting_points, epsilon=epsilon) - images

        return delta, is_adv

    def _initialization(self, images: torch.Tensor, labels: torch.Tensor):
        batch_size = len(images)
        delta = torch.zeros_like(images, device=self.device)
        epsilon = torch.full((batch_size,), float('inf'), device=self.device)
        is_adv = None

        if self.starting_points is not None:
            delta, is_adv = self._boundary_search(images, labels)

        if self.norm == 0:
            epsilon = torch.ones(batch_size, device=self.device) if self.starting_points is None else \
                                    delta.flatten(1).norm(p=0,dim=0)

        delta.requires_grad_()
        return epsilon, delta, is_adv

    def _init_optimizer(self, delta: torch.Tensor):
        if self.optimizer_config is None:
            optimizer = self.optimizer([delta], lr=self.alpha_init)
        else:
            if 'beta1' in self.optimizer_config:
                betas = (self.optimizer_config['beta1'], self.optimizer_config['beta2'])
                self.optimizer_config['betas'] = betas
                del self.optimizer_config['beta1']
                del self.optimizer_config['beta2']

            optimizer = self.optimizer([delta], **self.optimizer_config)

        return optimizer

    def _init_scheduler(self, optimizer: torch.optim, batch_size: int):
        scheduler = None

        if self.scheduler is not None:
            if self.scheduler_config is None:
                if issubclass(self.scheduler, CosineAnnealingLR):
                    scheduler = self.scheduler(optimizer, T_max=self.steps, eta_min=self.alpha_final)
                elif issubclass(self.scheduler, RLROPvec):
                    scheduler = self.scheduler(batch_size=batch_size, verbose=self.verbose, device=self.device)
                else:
                    scheduler = self.scheduler(optimizer, min_lr=self.alpha_final)
            elif not issubclass(self.scheduler, RLROPvec):
                scheduler = self.scheduler(optimizer, **self.scheduler_config)
            else:
                scheduler = self.scheduler(verbose=self.verbose, device=self.device, **self.scheduler_config)

        return scheduler

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        batch_size = len(images)

        dual, projection, _ = self._dual_projection_mid_points[self.norm]
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (images.ndim - 1))
        epsilon, delta, is_adv = self._initialization(images, labels)
        _worst_norm = torch.maximum(images, 1 - images).flatten(1).norm(p=self.norm, dim=1).detach()

        init_trackers = {
            'worst_norm': _worst_norm.to(self.device),
            'best_norm': _worst_norm.clone().to(self.device),
            'best_adv': adv_images,
            'adv_found': torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        }

        multiplier = 1 if self.targeted else -1

        # Instantiate Loss, Optimizer and Scheduler (if not None)
        if issubclass(self.loss, CE):
            loss_fn = self.loss(reduction='none')
        else:
            loss_fn = self.loss()

        optimizer = self._init_optimizer(delta)
        scheduler = self._init_scheduler(optimizer, batch_size)

        if scheduler is not None and isinstance(scheduler, RLROPvec):
            learning_rates = torch.ones(batch_size) * optimizer.param_groups[0]['lr']
            learning_rates = learning_rates.to(self.device)

        # Main Attack Loop
        for i in range(self.steps):
            optimizer.zero_grad()

            cosine = (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma = self.gamma_final + (self.gamma_init - self.gamma_final) * cosine

            delta_norm = delta.data.flatten(1).norm(p=self.norm, dim=1)
            adv_images = images + delta
            adv_images = adv_images.to(self.device)

            logits = self.model(adv_images)
            pred_labels = logits.argmax(dim=1)

            is_adv = (pred_labels == labels) if self.targeted else (pred_labels != labels)
            is_smaller = delta_norm < init_trackers['best_norm']
            is_both = is_adv & is_smaller
            init_trackers['adv_found'].logical_or_(is_adv)
            init_trackers['best_norm'] = torch.where(is_both, delta_norm, init_trackers['best_norm'])
            init_trackers['best_adv'] = torch.where(batch_view(is_both), adv_images.detach(),
                                                    init_trackers['best_adv'])

            if self.verbose:
                print(f"LR: {optimizer.param_groups[0]['lr']}")

            if self.norm == 0:
                epsilon = torch.where(is_adv,
                                      torch.minimum(torch.minimum(epsilon - 1, (epsilon * (1 - gamma)).floor_()),
                                                    init_trackers['best_norm']),
                                      torch.maximum(epsilon + 1, (epsilon * (1 + gamma)).floor_()))
                epsilon.clamp_(min=0)
            else:
                epsilon = torch.where(is_adv,
                                      torch.minimum(epsilon * (1 - gamma), init_trackers['best_norm']),
                                      torch.where(init_trackers['adv_found'],
                                                  epsilon * (1 + gamma),
                                                  float('inf'))
                                      )

            loss = loss_fn.forward(logits, labels)
            if isinstance(loss_fn, LL): loss = multiplier*loss

            if self.verbose:
                print(f"loss mean[{i}]:\n{loss.mean()}")
                print(f"steps[{i}]:\n{steps}")

            # Optimizer Step (gradient ascent)
            if isinstance(scheduler, RLROPvec):
                v_loss = torch.dot(loss, learning_rates)
                v_loss.backward()
            else:
                loss.sum().backward()

            # Clip Epsilon
            epsilon = torch.minimum(epsilon, init_trackers['worst_norm'])

            # Gradient Update
            delta.grad.data = torch.sign(delta.grad.data)
            optimizer.step()

            # Project In-place
            projection(delta=delta.data, epsilon=epsilon)
            # Clamp
            delta.data.add_(images).clamp_(min=0, max=1).sub_(images)

            # Scheduler Step
            if scheduler is not None and isinstance(scheduler, RLROPvec):
                steps = scheduler.step(loss, learning_rates)
            else:
                scheduler.step()

        return init_trackers['best_adv']
