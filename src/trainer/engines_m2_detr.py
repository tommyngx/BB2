import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from src.utils.bbox_loss import GIoULoss
from src.utils.m2_utils import compute_iou


class HungarianMatcher(nn.Module):
    """Hungarian algorithm for matching predictions to targets"""

    def __init__(self, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_bboxes, target_bboxes, target_mask):
        """
        Args:
            pred_bboxes: [B, N, 4]
            target_bboxes: [B, M, 4]
            target_mask: [B, M] - 1 for valid, 0 for padding
        Returns:
            List of (pred_idx, target_idx) for each batch
        """
        B, N, _ = pred_bboxes.shape
        _, M, _ = target_bboxes.shape

        # Flatten batch dimension
        pred_flat = pred_bboxes.flatten(0, 1)  # [B*N, 4]
        target_flat = target_bboxes.flatten(0, 1)  # [B*M, 4]

        # Compute L1 cost
        cost_bbox = torch.cdist(pred_flat, target_flat, p=1)  # [B*N, B*M]

        # Compute GIoU cost
        cost_giou = -compute_iou(pred_flat, target_flat)  # [B*N, B*M]

        # Total cost
        cost = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        cost = cost.view(B, N, -1, M).cpu()  # [B, N, B, M]

        # Match for each batch
        indices = []
        for i in range(B):
            # Get valid targets
            valid_mask = target_mask[i].bool()
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                indices.append(([], []))
                continue

            # Cost matrix for this batch: [N, num_valid]
            c = cost[i, :, i, valid_mask]

            # Hungarian matching
            pred_idx, target_idx = linear_sum_assignment(c)
            indices.append((pred_idx, target_idx))

        return indices


class DETRLoss(nn.Module):
    """DETR-style loss: classification + bbox + objectness"""

    def __init__(self, cls_criterion, lambda_bbox=5.0, lambda_giou=2.0, lambda_obj=1.0):
        super().__init__()
        self.cls_criterion = cls_criterion
        self.matcher = HungarianMatcher(cost_bbox=lambda_bbox, cost_giou=lambda_giou)
        self.bbox_criterion = nn.L1Loss(reduction="none")
        self.giou_loss = GIoULoss()
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou
        self.lambda_obj = lambda_obj

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with keys 'cls_logits', 'pred_bboxes', 'obj_scores'
            targets: dict with keys 'label', 'bboxes', 'bbox_mask'
        """
        # Classification loss
        cls_loss = self.cls_criterion(outputs["cls_logits"], targets["label"])

        # Detection loss (only for positive samples)
        pred_bboxes = outputs["pred_bboxes"]  # [B, N, 4]
        pred_obj_scores = outputs["obj_scores"]  # [B, N, 1]
        target_bboxes = targets["bboxes"]  # [B, M, 4]
        target_mask = targets["bbox_mask"]  # [B, M]

        # Hungarian matching
        indices = self.matcher(pred_bboxes, target_bboxes, target_mask)

        # Compute bbox loss for matched pairs
        bbox_loss = 0.0
        giou_loss = 0.0
        obj_loss = 0.0
        num_boxes = 0

        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # Matched predictions and targets
            matched_pred = pred_bboxes[i, pred_idx]
            matched_target = target_bboxes[i, target_idx]

            # L1 loss
            bbox_loss += self.bbox_criterion(matched_pred, matched_target).sum()

            # GIoU loss
            giou_loss += self.giou_loss(matched_pred, matched_target)

            num_boxes += len(pred_idx)

            # Objectness loss: matched = 1, unmatched = 0
            obj_target = torch.zeros(pred_bboxes.shape[1], 1, device=pred_bboxes.device)
            obj_target[pred_idx] = 1.0
            obj_loss += F.binary_cross_entropy_with_logits(
                pred_obj_scores[i], obj_target, reduction="sum"
            )

        # Average over boxes
        if num_boxes > 0:
            bbox_loss = bbox_loss / num_boxes
            giou_loss = giou_loss / num_boxes
            obj_loss = obj_loss / num_boxes

        # Total loss
        total_loss = (
            cls_loss
            + self.lambda_bbox * bbox_loss
            + self.lambda_giou * giou_loss
            + self.lambda_obj * obj_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss,
            "obj_loss": obj_loss,
        }
