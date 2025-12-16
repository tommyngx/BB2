import torch
import torch.nn as nn


class FocalBBoxLoss(nn.Module):
    """Focal loss for bounding box regression - focus on hard examples"""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, target):
        # Compute smooth L1 loss
        loss = self.smooth_l1(pred, target)

        # Compute focal weight
        pt = torch.exp(-loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply focal weight
        loss = focal_weight * loss
        return loss.mean()


class GIoULoss(nn.Module):
    """Generalized IoU Loss - better than L1 for bbox"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred, target: [N, 4] in format [x1, y1, x2, y2] normalized [0,1]
        # Compute IoU
        x1_i = torch.max(pred[:, 0], target[:, 0])
        y1_i = torch.max(pred[:, 1], target[:, 1])
        x2_i = torch.min(pred[:, 2], target[:, 2])
        y2_i = torch.min(pred[:, 3], target[:, 3])

        inter_w = torch.clamp(x2_i - x1_i, min=0)
        inter_h = torch.clamp(y2_i - y1_i, min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-6)

        # Compute enclosing box
        x1_c = torch.min(pred[:, 0], target[:, 0])
        y1_c = torch.min(pred[:, 1], target[:, 1])
        x2_c = torch.max(pred[:, 2], target[:, 2])
        y2_c = torch.max(pred[:, 3], target[:, 3])

        c_area = (x2_c - x1_c) * (y2_c - y1_c)

        # GIoU
        giou = iou - (c_area - union_area) / (c_area + 1e-6)

        # Loss = 1 - GIoU
        return (1 - giou).mean()
