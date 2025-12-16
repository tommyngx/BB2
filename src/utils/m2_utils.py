import torch
import math


def compute_iou(bbox_pred, bbox_true):
    """
    Compute IoU between predicted and true bboxes
    bbox format: [x1, y1, x2, y2] normalized to [0, 1]
    """
    # Get intersection coords
    x1_i = torch.max(bbox_pred[:, 0], bbox_true[:, 0])
    y1_i = torch.max(bbox_pred[:, 1], bbox_true[:, 1])
    x2_i = torch.min(bbox_pred[:, 2], bbox_true[:, 2])
    y2_i = torch.min(bbox_pred[:, 3], bbox_true[:, 3])

    # Intersection area
    inter_w = torch.clamp(x2_i - x1_i, min=0)
    inter_h = torch.clamp(y2_i - y1_i, min=0)
    inter_area = inter_w * inter_h

    # Union area
    pred_area = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (
        bbox_pred[:, 3] - bbox_pred[:, 1]
    )
    true_area = (bbox_true[:, 2] - bbox_true[:, 0]) * (
        bbox_true[:, 3] - bbox_true[:, 1]
    )
    union_area = pred_area + true_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def get_lambda_bbox_schedule(
    epoch, lambda_bbox_target=0.5, freeze_epochs=10, warmup_epochs=10
):
    """
    Lambda bbox schedule with cosine warmup (smoother than linear)

    - Epoch 0-9: lambda = 0 (chỉ học classification)
    - Epoch 10-19: lambda tăng dần từ 0 lên lambda_bbox_target (cosine)
    - Epoch 20+: lambda = lambda_bbox_target
    """
    if epoch < freeze_epochs:
        return 0.0
    elif epoch < freeze_epochs + warmup_epochs:
        # Cosine warmup instead of linear
        progress = (epoch - freeze_epochs) / warmup_epochs
        cosine_progress = (1 - math.cos(progress * math.pi)) / 2
        return lambda_bbox_target * cosine_progress
    else:
        return lambda_bbox_target
