import numpy as np

from ray import tune


OPTIMIZERS_SEARCH_TUNE = {
    'SGD': {
        'lr': tune.loguniform(1, 100),
        'momentum': tune.uniform(0.81, 0.99),
        'weight_decay': tune.loguniform(0.01, 1),
        'dampening': tune.uniform(0, 0.2)
    },
    'SGDNesterov': {
        'lr': tune.loguniform(1, 100),
        'momentum': tune.uniform(0.81, 0.99),
        'weight_decay': tune.loguniform(0.01, 1),
        'dampening': 0,
        'nesterov': True
    },
    'Adam':
    {
        'lr': tune.loguniform(1, 100),
        'weight_decay': tune.loguniform(0.01, 1),
        'eps': 1e-8,
        'amsgrad': False,
        'betas': (0.9, 0.999)
    },
    'AdamAmsgrad':
    {
        'lr': tune.loguniform(1, 100),
        'eps': 1e-8,
        'amsgrad': True,
        'betas': (0.9, 0.999)
    }
}

SCHEDULERS_SEARCH_TUNE = {
    'CosineAnnealingLR':
        {
            'T_max': lambda steps: steps,
            'eta_min': 0,
            'last_epoch': -1
        },
    'CosineAnnealingWarmRestarts':
        {
            'T_0': lambda steps: steps//2,
            'T_mult': 1,
            'eta_min': 0,
            'last_epoch': -1
        },
    'MultiStepLR':
        {
            'milestones': lambda steps: tune.grid_search(
                [tuple(np.linspace(0, steps, 10)),
                 tuple(np.linspace(0, steps, 5)),
                 tuple(np.linspace(0, steps, 3))]
            ),
            'gamma': tune.uniform(0.1, 0.9)
        },
    'ReduceLROnPlateau':
        {
            'factor': tune.uniform(0.1, 0.5),
            'patience': tune.choice([5, 10, 20]),
            'threshold': 1e-5
        }
}

