import torch


def compress_shift(X, alpha, warm_item_idx):
    item_warm_norms_mean = torch.norm(X[warm_item_idx], dim=1).mean()
    norms = torch.norm(X, dim=1, keepdim=True)
    norms_shifted = (norms + alpha * item_warm_norms_mean) / (alpha + 1)
    return X * norms_shifted / norms
