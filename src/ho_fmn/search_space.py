"""
Search Space definition for the Ax Tuner
"""

OPTIMIZER_PARAMS = {
    'SGD': {
        'lr': {"name": "lr", "type": "range", "bounds": [8/255, 10], "value_type": "float", "log_scale": True},
        'momentum': {"name": "momentum", "type": "range", "bounds": [0.0, 0.9], "value_type": "float"},
        'weight_decay': {"name": "weight_decay", "type": "range", "bounds": [0.01, 1.0], "value_type": "float"},
        'dampening': {"name": "dampening", "type": "range", "bounds": [0.0, 0.2], "value_type": "float"}
    },
    'Adam':
    {
        'lr': {"name": "lr", "type": "range", "bounds": [8/255, 10], "value_type": "float", "log_scale": True},
        'weight_decay': {"name": "weight_decay", "type": "range", "bounds": [0.01, 1.0], "value_type": "float"},
        'eps': {"name": "eps", "type": "fixed", "value": 1e-8, "value_type": "float"},
        'beta1': {"name": "beta1", "type": "range", "bounds": [0.0, 0.999], "value_type": "float"},
        'beta2': {"name": "beta2", "type": "range", "bounds": [0.0, 0.999], "value_type": "float"}
    },
    'Adamax':
    {
        'lr': {"name": "lr", "type": "range", "bounds": [8/255, 10], "value_type": "float", "log_scale": True},
        'weight_decay': {"name": "weight_decay", "type": "range", "bounds": [0.01, 1.0], "value_type": "float"},
        'eps': {"name": "eps", "type": "fixed", "value": 1e-8, "value_type": "float"},
        'beta1': {"name": "beta1", "type": "range", "bounds": [0.0, 0.999], "value_type": "float"},
        'beta2': {"name": "beta2", "type": "range", "bounds": [0.0, 0.999], "value_type": "float"}
    }
}

SCHEDULER_PARAMS = {
    'CALR': {
        'T_max': lambda steps: {"name": "T_max", "type": "fixed", "value": steps, "value_type": "int"},
        'eta_min':  {"name": "eta_min", "type": "fixed", "value": 0, "value_type": "int"},
        'last_epoch': {"name": "last_epoch", "type": "fixed", "value": -1, "value_type": "int"}
    },
    'RLROP':
    {
        'batch_size': lambda bs: {"name": "batch_size", "type": "fixed", "value": bs, "value_type": "int"},
        'factor': {"name": "factor", "type": "range", "bounds": [0.1, 0.5], "value_type": "float"},
        'patience': {"name": "patience", "type": "choice", "values": [2, 5, 10], "value_type": "int"},
        'threshold': {"name": "threshold", "type": "fixed", "value": 1e-4, "value_type": "float"},
        'eps': {"name": "eps", "type": "fixed", "value": 1e-8, "value_type": "float"}
    }
}

