import torch
from torch import Tensor

"""
Source: https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/attacks/fast_minimum_norm.py
"""

def simplex_projection(x, epsilon):
    """
    Simplex projection based on sorting.
    Parameters
    ----------
    x : Tensor
        Batch of vectors to project on the simplex.
    epsilon : float or Tensor
        Size of the simplex, default to 1 for the probability simplex.
    Returns
    -------
    projected_x : Tensor
        Batch of projected vectors on the simplex.
    """
    u = x.sort(dim=1, descending=True)[0]
    epsilon = epsilon.unsqueeze(1) if isinstance(epsilon, Tensor) else torch.tensor(epsilon, device=x.device)
    indices = torch.arange(x.size(1), device=x.device)
    cumsum = torch.cumsum(u, dim=1).sub_(epsilon).div_(indices + 1)
    k = (cumsum < u).long().mul_(indices).amax(dim=1, keepdim=True)
    tau = cumsum.gather(1, k)
    return (x - tau).clamp_(min=0)


def l1_ball_euclidean_projection(x, epsilon, inplace):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.
    Adapted from Tony Duan's implementation https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55

    Parameters
    ----------
    x: Tensor
        Batch of tensors to project.
    epsilon: float or Tensor
        Radius of L1-ball to project onto. Can be a single value for all tensors in the batch or a batch of values.
    inplace : bool
        Can optionally do the operation in-place.

    Returns
    -------
    projected_x: Tensor
        Batch of projected tensors with the same shape as x.

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    to_project = x.norm(p=1, dim=1) > epsilon
    if to_project.any():
        x_to_project = x[to_project]
        epsilon_ = epsilon[to_project] if isinstance(epsilon, Tensor) else torch.tensor([epsilon], device=x.device)
        if not inplace:
            x = x.clone()
        simplex_proj = simplex_projection(x_to_project.abs(), epsilon=epsilon_)
        x[to_project] = simplex_proj.copysign_(x_to_project)
        return x
    else:
        return x


def l0_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place l0 projection"""
    delta = delta.flatten(1)
    delta_abs = delta.abs()
    thresholds = delta_abs.topk(k=epsilon.long().max(), dim=1).values.gather(1, (epsilon.long().unsqueeze(1) - 1).clamp_(min=0))
    delta[delta_abs < thresholds] = 0
    return delta


def l1_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place l1 projection"""
    delta = l1_ball_euclidean_projection(x=delta.flatten(1), epsilon=epsilon, inplace=True)
    return delta


def l2_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place l2 projection"""
    delta = delta.flatten(1)
    l2_norms = delta.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-12)
    delta.mul_(epsilon.unsqueeze(1) / l2_norms).clamp_(max=1)
    return delta


def linf_projection_(delta: Tensor, epsilon: Tensor) -> Tensor:
    """In-place linf projection"""
    delta, epsilon = delta.flatten(1), epsilon.unsqueeze(1)
    delta = torch.clamp(delta, min=-epsilon, max=epsilon, out=delta)
    return delta


def l0_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    n_features = x0[0].numel()
    delta = l0_projection_(delta=x1 - x0, epsilon=n_features * epsilon)
    return delta.view_as(x0).add_(x0)


def l1_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    threshold = (1 - epsilon).unsqueeze(1)
    delta = (x1 - x0).flatten(1)
    delta_abs = delta.abs()
    mask = delta_abs <= threshold
    mid_points = delta_abs.sub_(threshold).copysign_(delta)
    mid_points[mask] = 0
    return mid_points.view_as(x0).add_(x0)


def l2_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    epsilon = epsilon.unsqueeze(1)
    return x0.flatten(1).mul(1 - epsilon).addcmul_(epsilon, x1.flatten(1)).view_as(x0)


def linf_mid_points(x0: Tensor, x1: Tensor, epsilon: Tensor) -> Tensor:
    epsilon = epsilon.unsqueeze(1)
    delta = (x1 - x0).flatten(1)
    delta = torch.clamp(delta, min=-epsilon, max=epsilon, out=delta)
    return delta.view_as(x0).add_(x0)


DUAL_PROJECTION_MIDPOINTS = {
    0: (None, l0_projection_, l0_mid_points),
    1: (float('inf'), l1_projection_, l1_mid_points),
    2: (2, l2_projection_, l2_mid_points),
    float('inf'): (1, linf_projection_, linf_mid_points),
}

