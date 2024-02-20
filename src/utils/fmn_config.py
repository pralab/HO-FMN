from src.utils.loss import LL, DLR, CE
from torch.optim import SGD, Adam, Adamax
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.schedulers.RLROP_vec import ReduceLROnPlateau as RLROP

LOSSES = {
    'LL': LL,
    'DLR': DLR,
    'CE': CE
}

OPTIMIZERS = {
    'SGD': SGD,
    'Adam': Adam,
    'Adamax': Adamax
}

SCHEDULERS = {
    'CALR': CosineAnnealingLR,
    'RLROP': RLROP,
    None: None
}


def print_fmn_configs():
    print("\nLosses:")
    for loss in LOSSES.keys():
        print(f"\t{loss}")

    print("Optimizers:")
    for optimizer in OPTIMIZERS.keys():
        print(f"\t{optimizer}")

    print("Schedulers:")
    for scheduler in SCHEDULERS.keys():
        if scheduler is not None: print(f"\t{scheduler}")