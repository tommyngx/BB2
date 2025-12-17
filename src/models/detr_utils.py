import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    Args:
        boxes1: [N, 4] in format [x1, y1, x2, y2]
        boxes2: [M, 4] in format [x1, y1, x2, y2]
    Returns:
        iou: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def hungarian_matching(pred_boxes, pred_scores, gt_boxes, cost_bbox=5.0, cost_giou=2.0):
    """
    Hungarian matching between predictions and ground truths
    Args:
        pred_boxes: [N_q, 4]
        pred_scores: [N_q, 1]
        gt_boxes: [N_gt, 4]
    Returns:
        indices: (pred_idx, gt_idx) matched pairs
    """
    N_gt = gt_boxes.shape[0]

    if N_gt == 0:
        return [], []

    # L1 cost
    cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)

    # IoU cost (negative for maximization)
    iou = box_iou(pred_boxes, gt_boxes)
    cost_giou = -iou

    # Total cost
    C = cost_bbox * cost_bbox + cost_giou * cost_giou
    C = C.cpu().detach().numpy()

    # Hungarian algorithm
    pred_idx, gt_idx = linear_sum_assignment(C)

    return pred_idx.tolist(), gt_idx.tolist()


def compute_iou_weighted_obj_loss(pred_boxes, pred_scores, gt_boxes, dn_outputs=None):
    """
    Compute IoU-weighted objectness loss
    Args:
        pred_boxes: [B, N_q, 4]
        pred_scores: [B, N_q, 1]
        gt_boxes: [B, N_gt, 4]
        dn_outputs: list of denoising outputs
    Returns:
        loss: scalar
    """
    B = pred_boxes.shape[0]
    total_loss = 0.0

    for b in range(B):
        pred_b = pred_boxes[b]
        score_b = pred_scores[b]
        gt_b = gt_boxes[b]

        # Remove padding (boxes with all zeros)
        valid_gt = gt_b.sum(dim=-1) > 0
        gt_b = gt_b[valid_gt]

        if gt_b.shape[0] == 0:
            target = torch.zeros_like(score_b)
            weight = torch.ones_like(score_b)
            loss_b = F.binary_cross_entropy_with_logits(score_b, target, weight=weight)
            total_loss += loss_b
            continue

        # Hungarian matching
        pred_idx, gt_idx = hungarian_matching(pred_b, score_b, gt_b)

        # Compute IoU for matched pairs
        target = torch.zeros_like(score_b)
        weight = torch.ones_like(score_b)

        for p_idx, g_idx in zip(pred_idx, gt_idx):
            iou = box_iou(pred_b[p_idx : p_idx + 1], gt_b[g_idx : g_idx + 1])[0, 0]
            target[p_idx] = 1.0
            weight[p_idx] = iou.clamp(min=0.01)

        loss_b = F.binary_cross_entropy_with_logits(score_b, target, weight=weight)
        total_loss += loss_b

    # Add denoising loss
    if dn_outputs is not None and len(dn_outputs) > 0:
        for dn_output in dn_outputs:
            dn_pred_boxes = dn_output["pred_bboxes"]
            dn_pred_scores = dn_output["obj_scores"]
            dn_gt_boxes = dn_output["gt_boxes"]

            for b in range(B):
                dn_pred_b = dn_pred_boxes[b]
                dn_score_b = dn_pred_scores[b]
                dn_gt_b = dn_gt_boxes[b]

                valid_gt = dn_gt_b.sum(dim=-1) > 0
                dn_gt_b = dn_gt_b[valid_gt]
                dn_pred_b = dn_pred_b[valid_gt]
                dn_score_b = dn_score_b[valid_gt]

                if dn_gt_b.shape[0] == 0:
                    continue

                iou = box_iou(dn_pred_b, dn_gt_b).diagonal()
                target = torch.ones_like(dn_score_b)
                weight = iou.unsqueeze(1).clamp(min=0.01)

                loss_dn = F.binary_cross_entropy_with_logits(
                    dn_score_b, target, weight=weight
                )
                total_loss += loss_dn

    return total_loss / B
