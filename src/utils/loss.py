import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss as CE


class LL(nn.Module):
    def __init__(self):
        super(LL, self).__init__()

    def _difference_of_logits(self, logits, labels):
        labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))

        class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = (logits - labels_infhot).amax(dim=1)

        return class_logits - other_logits

    def forward(self, inputs, targets):
        loss = self._difference_of_logits(inputs, targets)
        return loss


class DLR(nn.Module):
    """
    DLR loss, original implementation by Croce et Al.
    Source: https://github.com/fra31/auto-attack
    """
    def __init__(self):
        super(DLR, self).__init__()

    def _difference_of_logits_ratio(self, logits, labels):
        logits_sorted, ind_sorted = logits.sort(dim=1)
        ind = (ind_sorted[:, -1] == labels).float()
        u = torch.arange(logits.shape[0])

        return -(logits[u, labels] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (
                1. - ind)) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)

    def forward(self, inputs, targets):
        loss = self._difference_of_logits_ratio(inputs, targets)
        return loss


