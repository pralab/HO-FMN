from src.utils.loss import LL, DLR, CE
from torch.optim import SGD, Adam, Adamax
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.schedulers.RLROP_vec import ReduceLROnPlateau as RLROPvec

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
    'RLROPVec': RLROPvec,
    None: None
}
