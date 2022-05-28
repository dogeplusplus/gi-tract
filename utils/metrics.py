import torch
import typing as t


def dice_coef(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    thr: float = 0.5,
    dim: t.Tuple[int, int] = (2, 3),
    epsilon: float = 1e-3,
) -> torch.Tensor:
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    thr: float = 0.5,
    dim: t.Tuple[int, int] = (2, 3),
    epsilon: float = 1e-3,
) -> torch.Tensor:
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou
