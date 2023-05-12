import math

import torch
from torch import nn, Tensor
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import \
    CosineAnnealingLR, \
    CosineAnnealingWarmRestarts, \
    MultiStepLR, \
    ReduceLROnPlateau

import numpy as np

from functools import partial
from typing import Optional, Union


def linf_projection_(delta, epsilon):
    """In-place linf projection"""
    delta = delta.flatten(1)
    epsilon = epsilon.unsqueeze(1)
    torch.maximum(torch.minimum(delta, epsilon, out=delta), -epsilon, out=delta)


def linf_mid_points(x0, x1, epsilon):
    epsilon = epsilon.unsqueeze(1)
    delta = (x1 - x0).flatten(1)
    return x0 + torch.maximum(torch.minimum(delta, epsilon, out=delta), -epsilon, out=delta).view_as(x0)


def difference_of_logits(logits, labels, labels_infhot = None):
    if labels_infhot is None:
        labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).amax(dim=1)
    return class_logits - other_logits


class FMNOpt:
    def __init__(self,
                 model: nn.Module,
                 inputs: Tensor,
                 labels: Tensor,
                 norm: Union[str, int],
                 targeted: bool = False,
                 steps: int = 10,
                 gamma_init: float = 0.05,
                 gamma_final: float = 0.001,
                 starting_points: Optional[Tensor] = None,
                 binary_search_steps: int = 10,
                 optimizer='SGD',
                 scheduler='CosineAnnealingLR',
                 optimizer_config=None,
                 scheduler_config=None,
                 device=torch.device('cpu'),
                 logit_loss = True
                 ):
        self.model = model
        self.inputs = inputs.to(device)
        self.labels = labels.to(device)
        self.norm = float('inf') if norm == 'inf' else int(norm)
        self.targeted = targeted
        self.steps = steps
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.starting_points = starting_points
        self.binary_search_steps = binary_search_steps
        self.device = device
        self.batch_size = len(self.inputs)
        self.batch_view = lambda tensor: tensor.view(self.batch_size, *[1] * (self.inputs.ndim - 1))
        self.optimizer_config=optimizer_config
        self.scheduler_config=scheduler_config
        self.logit_loss = logit_loss

        self._dual_projection_mid_points = {
            float('inf'): (1, linf_projection_, linf_mid_points),
        }

        _worst_norm = torch.maximum(inputs, 1 - inputs).flatten(1).norm(p=self.norm, dim=1)
        self.init_trackers = {
            'worst_norm': _worst_norm.to(self.device),
            'best_norm': _worst_norm.clone().to(self.device),
            'best_adv': self.inputs.clone().to(self.device),
            'adv_found': torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        }

        self._optimizers = {
            "SGD": SGD,
            'SGDNesterov': SGD,
            "Adam": Adam,
            'AdamAmsgrad': Adam
        }
        self._schedulers = {
            "CosineAnnealingLR": CosineAnnealingLR,
            "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
            "MultiStepLR": MultiStepLR,
            "ReduceLROnPlateau": ReduceLROnPlateau
        }

        self.optimizer_name = optimizer
        self.scheduler_name = scheduler

        self.optimizer = None
        self.scheduler = None

    def _boundary_search(self):
        _, _, mid_point = self._dual_projection_mid_points[self.norm]

        is_adv = self.model(self.starting_points).argmax(dim=1)
        if not is_adv.all():
            raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(self.batch_size, device=self.device)
        upper_bound = torch.ones(self.batch_size, device=self.device)
        for _ in range(self.binary_search_steps):
            epsilon = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=self.inputs, x1=self.starting_points, epsilon=epsilon)
            pred_labels = self.model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == self.labels) if self.targeted else (pred_labels != self.labels)
            lower_bound = torch.where(is_adv, lower_bound, epsilon)
            upper_bound = torch.where(is_adv, epsilon, upper_bound)

        delta = mid_point(x0=self.inputs, x1=self.starting_points, epsilon=epsilon) - self.inputs

        return epsilon, delta, is_adv

    def _init_attack(self):
        delta = torch.zeros_like(self.inputs, device=self.device)
        is_adv = None

        if self.starting_points is not None:
            epsilon, delta, is_adv = self._boundary_search()

        if self.norm == 0:
            epsilon = torch.ones(self.batch_size,
                                 device=self.device) if self.starting_points is None else delta.flatten(1).norm(p=0,
                                                                                                                dim=0)
        else:
            epsilon = torch.full((self.batch_size,), float('inf'), device=self.device)

        return epsilon, delta, is_adv

    def _init_optimizer(self, objective=None):
        assert objective is not None

        optimizer = self._optimizers[self.optimizer_name]

        if isinstance(optimizer, list) and len(optimizer) > 0:
            opt_params = optimizer[1]
            optimizer = optimizer[0]([objective], **self.optimizer_config, **opt_params)
        else:
            optimizer = optimizer([objective], **self.optimizer_config)

        self.optimizer = optimizer

    def _init_scheduler(self):
        assert self.optimizer is not None

        scheduler = self._schedulers[self.scheduler_name]

        if scheduler == 'MultiStepLR':
            milestones = len(self.scheduler_config['milestones'])
            self.scheduler_config['milestones'] = np.linspace(0, self.steps, milestones)

        if scheduler == 'CosineAnnealingLR':
            self.scheduler_config['T_max'] = self.steps

        if scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler_config['T_0'] = self.steps//2

        if isinstance(scheduler, list) and len(scheduler) > 0:
            sch_params = scheduler[1]
            scheduler = scheduler[0](self.optimizer, **self.scheduler_config, **sch_params)
        else:
            scheduler = scheduler(self.optimizer, **self.scheduler_config)

        self.scheduler = scheduler

    def _scheduler_step(self, *step_params):
        assert self.scheduler is not None

        self.scheduler.step(*step_params)

    def run(self, log=False):
        dual, projection, _ = self._dual_projection_mid_points[self.norm]

        epsilon, delta, is_adv = self._init_attack()
        multiplier = 1 if self.targeted else -1

        delta.requires_grad_(True)

        # Initialize optimizer and scheduler
        self._init_optimizer(objective=delta)
        self._init_scheduler()

        if log:
            print("Starting the attack...\n")
        for i in range(self.steps):
            if log:
                print(f"Attack completion: {i / self.steps * 100:.2f}%")
            self.optimizer.zero_grad()

            cosine = (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma = self.gamma_final + (self.gamma_init - self.gamma_final) * cosine

            delta_norm = delta.data.flatten(1).norm(p=self.norm, dim=1)
            adv_inputs = self.inputs + delta

            logits = self.model(adv_inputs)
            pred_labels = logits.argmax(dim=1)

            if self.logit_loss:
                if i == 0:
                    labels_infhot = torch.zeros_like(logits).scatter_(1, self.labels.unsqueeze(1), float('inf'))
                    logit_diff_func = partial(difference_of_logits, labels=self.labels, labels_infhot=labels_infhot)

                logit_diffs = logit_diff_func(logits=logits)
                loss = -(multiplier * logit_diffs)

            else:
                c_loss = nn.CrossEntropyLoss()
                loss = -c_loss(logits, self.labels)

            loss.sum().backward()
            delta_grad = delta.grad.data

            is_adv = (pred_labels == self.labels) if self.targeted else (pred_labels != self.labels)
            is_smaller = delta_norm < self.init_trackers['best_norm']
            is_both = is_adv & is_smaller
            self.init_trackers['adv_found'].logical_or_(is_adv)
            self.init_trackers['best_norm'] = torch.where(is_both, delta_norm, self.init_trackers['best_norm'])
            self.init_trackers['best_adv'] = torch.where(self.batch_view(is_both), adv_inputs.detach(),
                                                         self.init_trackers['best_adv'])

            if self.norm == 0:
                epsilon = torch.where(is_adv,
                                      torch.minimum(torch.minimum(epsilon - 1,
                                                                  (epsilon * (1 - gamma)).floor_()),
                                                    self.init_trackers['best_norm']),
                                      torch.maximum(epsilon + 1, (epsilon * (1 + gamma)).floor_()))
                epsilon.clamp_(min=0)
            else:
                distance_to_boundary = loss.detach().abs() / delta_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)
                epsilon = torch.where(is_adv,
                                      torch.minimum(epsilon * (1 - gamma), self.init_trackers['best_norm']),
                                      torch.where(self.init_trackers['adv_found'],
                                                  epsilon * (1 + gamma),
                                                  delta_norm + distance_to_boundary)
                                      )

            # clip epsilon
            epsilon = torch.minimum(epsilon, self.init_trackers['worst_norm'])

            # normalize gradient
            grad_l2_norms = delta_grad.flatten(1).norm(p=2, dim=1).clamp_(min=1e-12)
            delta_grad.div_(self.batch_view(grad_l2_norms))

            self.optimizer.step()

            # project in place
            projection(delta=delta.data, epsilon=epsilon)

            # clamp
            delta.data.add_(self.inputs).clamp_(min=0, max=1).sub_(self.inputs)

            # Computing the best distance (x-x0 for the adversarial)
            _distance = torch.linalg.norm((self.init_trackers['best_adv'] - self.inputs).data.flatten(1),
                                          dim=1, ord=self.norm)

            if self.scheduler_name == 'ReduceLROnPlateau':
                self._scheduler_step(torch.median(_distance).item())
            else:
                self._scheduler_step()

        if log:
            print("Attack completed!\n")

        return torch.median(_distance), self.init_trackers['best_adv']