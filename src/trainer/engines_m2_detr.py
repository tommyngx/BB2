import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import csv
from datetime import datetime
from scipy.optimize import linear_sum_assignment

from src.utils.loss import FocalLoss, LDAMLoss, FocalLoss2
from src.utils.plot import plot_metrics
from src.utils.bbox_loss import GIoULoss
from src.utils.m2_utils import compute_iou


class HungarianMatcher(nn.Module):
    """Hungarian algorithm for bipartite matching between predictions and targets"""

    def __init__(self, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_bboxes, target_bboxes, target_mask):
        """
        Args:
            pred_bboxes: [B, N, 4] predicted boxes
            target_bboxes: [B, M, 4] target boxes
            target_mask: [B, M] 1 for valid, 0 for padding
        Returns:
            List of (pred_idx, target_idx) for each sample
        """
        B, N = pred_bboxes.shape[:2]
        M = target_bboxes.shape[1]

        indices = []
        for i in range(B):
            # Get valid targets FIRST
            valid_mask = target_mask[i] > 0.5
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                indices.append(([], []))
                continue

            # Extract ONLY valid targets
            pred = pred_bboxes[i]  # [N, 4]
            tgt = target_bboxes[i][valid_mask]  # [num_valid, 4]

            # Compute costs on [N, num_valid] matrix
            try:
                # L1 cost: [N, num_valid]
                cost_bbox = torch.cdist(pred, tgt, p=1)

                # GIoU cost: [N, num_valid]
                cost_giou = -compute_iou(pred, tgt)

                # Total cost: [N, num_valid]
                C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
                C = C.cpu().numpy()

                # CRITICAL FIX: Ensure C is exactly [N, num_valid]
                assert C.shape == (N, num_valid), (
                    f"Cost matrix shape mismatch: {C.shape} != ({N}, {num_valid})"
                )

                # Hungarian matching: returns indices in range [0, N-1] and [0, num_valid-1]
                pred_idx, tgt_idx = linear_sum_assignment(C)

                # CRITICAL VALIDATION
                if len(pred_idx) > 0:
                    max_pred = max(pred_idx)
                    max_tgt = max(tgt_idx)
                    if max_pred >= N or max_tgt >= num_valid:
                        raise ValueError(
                            f"Invalid indices from Hungarian: pred_idx max={max_pred} (N={N}), "
                            f"tgt_idx max={max_tgt} (num_valid={num_valid})"
                        )

                indices.append((pred_idx.tolist(), tgt_idx.tolist()))

            except Exception as e:
                print(f"⚠️ Error in Hungarian matching for sample {i}: {e}")
                print(f"  pred shape: {pred.shape}, tgt shape: {tgt.shape}")
                print(f"  Cost matrix shape: {C.shape if 'C' in locals() else 'N/A'}")
                print(f"  Expected: ({N}, {num_valid})")
                indices.append(([], []))

        return indices


class DETRCriterion(nn.Module):
    """
    DETR loss with classification + bbox regression + objectness
    Supports auxiliary losses from intermediate decoder layers
    """

    def __init__(
        self,
        cls_criterion,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        lambda_obj=1.0,
        aux_loss_weight=0.5,
    ):
        super().__init__()
        self.cls_criterion = cls_criterion
        self.matcher = HungarianMatcher(cost_bbox=lambda_bbox, cost_giou=lambda_giou)
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou
        self.lambda_obj = lambda_obj
        self.aux_loss_weight = aux_loss_weight

        self.bbox_l1_loss = nn.L1Loss(reduction="none")
        self.giou_loss = GIoULoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'cls_logits', 'pred_bboxes', 'obj_scores', 'aux_outputs'
            targets: dict with 'label', 'bboxes', 'bbox_mask'
        """
        # Classification loss (per image, not per query)
        cls_loss = self.cls_criterion(outputs["cls_logits"], targets["label"])

        # Get predictions and targets
        pred_bboxes = outputs["pred_bboxes"]  # [B, N, 4]
        pred_obj = outputs["obj_scores"]  # [B, N, 1]
        target_bboxes = targets["bboxes"]  # [B, M, 4]
        target_mask = targets["bbox_mask"]  # [B, M]

        # Hungarian matching
        indices = self.matcher(pred_bboxes, target_bboxes, target_mask)

        # Compute bbox and objectness losses
        bbox_loss, giou_loss, obj_loss = self._compute_detection_losses(
            pred_bboxes, pred_obj, target_bboxes, target_mask, indices
        )

        # Total loss for main predictions
        total_loss = (
            cls_loss
            + self.lambda_bbox * bbox_loss
            + self.lambda_giou * giou_loss
            + self.lambda_obj * obj_loss
        )

        # Auxiliary losses from intermediate decoder layers
        aux_loss = 0.0
        if "aux_outputs" in outputs and len(outputs["aux_outputs"]) > 0:
            for aux_out in outputs["aux_outputs"]:
                aux_bbox = aux_out["pred_bboxes"]
                aux_obj = aux_out["obj_scores"]

                # Reuse same matching
                aux_bbox_loss, aux_giou_loss, aux_obj_loss = (
                    self._compute_detection_losses(
                        aux_bbox, aux_obj, target_bboxes, target_mask, indices
                    )
                )
                aux_loss += (
                    self.lambda_bbox * aux_bbox_loss
                    + self.lambda_giou * aux_giou_loss
                    + self.lambda_obj * aux_obj_loss
                )

            aux_loss = aux_loss * self.aux_loss_weight / len(outputs["aux_outputs"])
            total_loss = total_loss + aux_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss,
            "obj_loss": obj_loss,
            "aux_loss": aux_loss
            if isinstance(aux_loss, torch.Tensor)
            else torch.tensor(0.0),
        }

    def _compute_detection_losses(
        self, pred_bboxes, pred_obj, target_bboxes, target_mask, indices
    ):
        """Compute bbox, GIoU, and objectness losses given matched indices"""
        B, N = pred_bboxes.shape[:2]
        device = pred_bboxes.device

        bbox_loss = torch.tensor(0.0, device=device)
        giou_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        num_boxes = 0

        for i, (pred_idx, tgt_idx) in enumerate(indices):
            obj_target = torch.zeros(N, 1, device=device)

            if len(pred_idx) == 0:
                obj_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_obj[i], obj_target, reduction="sum"
                )
                continue

            # Get valid targets
            valid_mask = target_mask[i] > 0.5
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                obj_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_obj[i], obj_target, reduction="sum"
                )
                continue

            valid_targets = target_bboxes[i][valid_mask]  # [num_valid, 4]

            # Sanity checks before indexing
            try:
                assert len(pred_idx) <= N, f"pred_idx length {len(pred_idx)} > N {N}"
                assert len(tgt_idx) <= num_valid, (
                    f"tgt_idx length {len(tgt_idx)} > num_valid {num_valid}"
                )
                assert max(pred_idx) < N, f"max(pred_idx)={max(pred_idx)} >= N={N}"
                assert max(tgt_idx) < num_valid, (
                    f"max(tgt_idx)={max(tgt_idx)} >= num_valid={num_valid}"
                )

                matched_pred = pred_bboxes[i, pred_idx]
                matched_tgt = valid_targets[tgt_idx]
            except Exception as e:
                print(f"⚠️ Indexing error at sample {i}: {e}")
                print(
                    f"  pred_idx: {pred_idx}, max: {max(pred_idx) if pred_idx else 'N/A'}"
                )
                print(
                    f"  tgt_idx: {tgt_idx}, max: {max(tgt_idx) if tgt_idx else 'N/A'}"
                )
                print(f"  N={N}, num_valid={num_valid}")
                obj_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_obj[i], obj_target, reduction="sum"
                )
                continue

            # Validate bbox values
            invalid_pred = (matched_pred < 0).any() | (matched_pred > 1).any()
            invalid_tgt = (
                (matched_tgt < 0).any()
                | (matched_tgt > 1).any()
                | (matched_tgt[:, 2] <= 0).any()
                | (matched_tgt[:, 3] <= 0).any()
            )

            if invalid_pred or invalid_tgt:
                print(f"⚠️ Invalid bbox at sample {i}, skipping")
                obj_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_obj[i], obj_target, reduction="sum"
                )
                continue

            # L1 loss
            bbox_loss += self.bbox_l1_loss(matched_pred, matched_tgt).sum()

            # GIoU loss
            try:
                giou_loss += self.giou_loss(matched_pred, matched_tgt)
            except Exception as e:
                print(f"⚠️ GIoU error: {e}")

            # Objectness
            obj_target[pred_idx] = 1.0
            obj_loss += nn.functional.binary_cross_entropy_with_logits(
                pred_obj[i], obj_target, reduction="sum"
            )

            num_boxes += len(pred_idx)

        # Normalize
        if num_boxes > 0:
            bbox_loss = bbox_loss / num_boxes
            giou_loss = giou_loss / num_boxes
            obj_loss = obj_loss / (B * N)
        else:
            obj_loss = obj_loss / (B * N)

        return bbox_loss, giou_loss, obj_loss


def train_m2_detr_model(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    lr=1e-4,
    device="cpu",
    model_name="m2_detr",
    output="output",
    dataset_folder="None",
    train_df=None,
    patience=50,
    loss_type="ce",
    lambda_bbox=5.0,
    lambda_giou=2.0,
    lambda_obj=1.0,
):
    """Train M2 DETR model with auxiliary losses"""
    # Option 1: Single GPU (recommended for now)
    model = model.to(device)

    # Option 2: Multi-GPU with DistributedDataParallel (advanced)
    # if device != "cpu" and torch.cuda.device_count() > 1:
    #     import torch.distributed as dist
    #     from torch.nn.parallel import DistributedDataParallel as DDP
    #
    #     # Initialize process group
    #     dist.init_process_group(backend="nccl")
    #     model = DDP(model, device_ids=[device])

    # Classification loss
    if train_df is not None:
        class_counts = train_df["cancer"].value_counts()
        total_samples = len(train_df)
        num_classes = len(class_counts)
        weights = torch.tensor(
            [
                total_samples / (num_classes * class_counts[i])
                for i in range(num_classes)
            ],
            dtype=torch.float,
        ).to(device)
    else:
        weights = None

    if loss_type == "focal":
        cls_criterion = FocalLoss(alpha=weights)
        print("Using Focal Loss")
    elif loss_type == "ldam":
        cls_criterion = LDAMLoss(
            cls_num_list=train_df["cancer"].value_counts().sort_index().tolist(),
            weight=weights,
        ).to(device)
        print("Using LDAM Loss")
    elif loss_type == "focal2":
        cls_criterion = FocalLoss2(alpha=weights)
        print("Using Focal Loss v2")
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=weights)
        print(
            "Using CrossEntropyLoss" + (" (with weight)" if weights is not None else "")
        )

    # DETR criterion
    criterion = DETRCriterion(
        cls_criterion,
        lambda_bbox=lambda_bbox,
        lambda_giou=lambda_giou,
        lambda_obj=lambda_obj,
        aux_loss_weight=0.5,
    )

    # Optimizer with different LR for backbone and decoder
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": lr * 0.1,
        },
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)
    scaler = GradScaler() if device != "cpu" else None
    print(f"Using AMP: {scaler is not None}")

    # LR schedulers
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.95 ** (epoch - warmup_epochs)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    # Setup directories
    model_dir = os.path.join(output, "models")
    plot_dir = os.path.join(output, "figures")
    log_dir = os.path.join(output, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = dataset_folder.split("/")[-1]
    try:
        sample = next(iter(train_loader))
        img_size = (sample["image"].shape[2], sample["image"].shape[3])
    except:
        img_size = (448, 448)

    model_key = f"{dataset}_{img_size[0]}x{img_size[1]}_detr_{model_name}"
    print(f"Model key: {model_key}")

    # Check for existing weights
    print(f"Checking for existing weights in {model_dir}\n with model_key: {model_key}")
    existing_weights = []
    for fname in os.listdir(model_dir):
        if fname.startswith(model_key) and fname.endswith(".pth"):
            try:
                acc_part = fname.replace(".pth", "").split("_")[-1]
                acc_val = float(acc_part) / 10000
                existing_weights.append((fname, acc_val))
            except Exception:
                print(f"Skipping invalid model file: {fname}")
    if existing_weights:
        print("Found existing weights:")
        for fname, acc_val in existing_weights:
            print(f"  - {fname} (accuracy: {acc_val:.4f})")
    else:
        print("No existing weights found.")

    # Training history
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    train_ious, test_ious = [], []
    best_acc = 0.0
    patience_counter = 0
    last_lr = lr
    skip_save_epochs = 10  # Skip saving for first 10 epochs

    log_file = os.path.join(log_dir, f"{model_key}.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "datetime",
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_iou",
                    "test_loss",
                    "test_acc",
                    "test_iou",
                    "test_map",
                    "lr",
                    "patience",
                    "best_acc",
                ]
            )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        epoch_iou = 0.0
        num_bbox_samples = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for batch in loop:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bboxes"].to(device)  # [B, M, 4]
            bbox_mask = batch["bbox_mask"].to(device)  # [B, M]

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    targets = {
                        "label": labels,
                        "bboxes": bboxes,
                        "bbox_mask": bbox_mask,
                    }
                    loss_dict = criterion(outputs, targets)
                    loss = loss_dict["total_loss"]

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                targets = {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask}
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Classification accuracy
            _, predicted = torch.max(outputs["cls_logits"], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Compute IoU for training
            if bbox_mask.sum() > 0:
                with torch.no_grad():
                    pred_bboxes = outputs["pred_bboxes"]
                    for i in range(images.size(0)):
                        valid_mask = bbox_mask[i] > 0.5
                        if valid_mask.sum() > 0:
                            # Use first valid bbox for simplicity
                            pred = pred_bboxes[i, 0:1]  # [1, 4]
                            tgt = bboxes[i][valid_mask][:1]  # [1, 4]
                            iou = compute_iou(pred, tgt)
                            epoch_iou += iou.item()
                            num_bbox_samples += 1

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        avg_train_iou = epoch_iou / max(num_bbox_samples, 1)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        train_ious.append(avg_train_iou)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
            f"IoU: {avg_train_iou:.4f} Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ✅ ENHANCED VALIDATION WITH IoU AND mAP
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        val_iou = 0.0
        num_val_bbox = 0

        # For detailed metrics
        all_preds = []
        all_labels = []
        all_probs = []

        # For mAP computation
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bboxes"].to(device)
                bbox_mask = batch["bbox_mask"].to(device)

                outputs = model(images)
                targets = {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask}
                loss_dict = criterion(outputs, targets)

                val_loss += loss_dict["total_loss"].item() * images.size(0)

                # Get predictions and probabilities
                probs = torch.softmax(outputs["cls_logits"], dim=1)
                _, predicted = torch.max(outputs["cls_logits"], 1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Store for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Compute IoU and collect boxes for mAP
                pred_bboxes = outputs["pred_bboxes"]
                pred_obj = torch.sigmoid(outputs["obj_scores"])  # [B, N, 1]

                for i in range(images.size(0)):
                    valid_mask = bbox_mask[i] > 0.5
                    if valid_mask.sum() > 0:
                        # Get top prediction by objectness score
                        top_idx = pred_obj[i].squeeze(-1).argmax()
                        pred_box = pred_bboxes[i, top_idx : top_idx + 1]  # [1, 4]
                        pred_score = pred_obj[i, top_idx].item()

                        gt_boxes = bboxes[i][valid_mask]  # [K, 4]

                        # Compute IoU
                        iou = compute_iou(pred_box, gt_boxes[:1])  # Use first GT
                        val_iou += iou.item()
                        num_val_bbox += 1

                        # Store for mAP
                        all_pred_boxes.append(pred_box.cpu())
                        all_pred_scores.append(pred_score)
                        all_gt_boxes.append(gt_boxes.cpu())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        avg_val_iou = val_iou / max(num_val_bbox, 1)
        test_losses.append(val_loss)
        test_accs.append(val_acc)
        test_ious.append(avg_val_iou)

        # Compute mAP@0.5
        val_map = 0.0
        if len(all_pred_boxes) > 0:
            num_correct = 0
            for pred_box, gt_box in zip(all_pred_boxes, all_gt_boxes):
                iou = compute_iou(pred_box, gt_box[:1])
                if iou.item() >= 0.5:
                    num_correct += 1
            val_map = num_correct / len(all_pred_boxes)

        # ✅ COMPUTE DETAILED METRICS
        from sklearn.metrics import (
            classification_report,
            roc_auc_score,
            precision_recall_fscore_support,
        )
        import numpy as np

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        # AUC (only for binary classification)
        try:
            if all_probs.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1]) * 100
            else:
                auc = 0.0
        except:
            auc = 0.0

        # Sensitivity (Recall for positive class)
        if len(np.unique(all_labels)) > 1:
            _, recall_per_class, _, _ = precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0
            )
            sensitivity = (
                recall_per_class[1] * 100 if len(recall_per_class) > 1 else 0.0
            )
        else:
            sensitivity = 0.0

        # ✅ PRINT METRICS
        print(
            f"Test Accuracy : {val_acc * 100:.2f}% | Loss: {val_loss:.4f} | AUC: {auc:.2f}%"
        )
        print(f"Test Precision: {precision * 100:.2f}% | Sens: {sensitivity:.2f}%")
        print(
            f"Test IoU      : {avg_val_iou * 100:.2f}% | mAP@0.5: {val_map * 100:.2f}%"
        )

        # Print classification report
        print(classification_report(all_labels, all_preds, zero_division=0))

        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    epoch + 1,
                    round(epoch_loss, 6),
                    round(epoch_acc, 6),
                    round(avg_train_iou, 6),
                    round(val_loss, 6),
                    round(val_acc, 6),
                    round(avg_val_iou, 6),
                    round(val_map, 6),
                    optimizer.param_groups[0]["lr"],
                    patience_counter,
                    round(best_acc, 6),
                ]
            )

        # LR scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < last_lr:
            print(
                f"Learning rate reduced to {current_lr:.6f}, resetting patience counter"
            )
            patience_counter = 0
            last_lr = current_lr
        else:
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                datetime.now().isoformat(),
                                f"early_stop_epoch_{epoch + 1}",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
                    break

        # ✅ TOP-2 MODEL SAVING (SKIP FIRST 10 EPOCHS)
        if epoch < skip_save_epochs:
            print(
                f"⏸️ Skipping model save (warmup period, epoch {epoch + 1}/{skip_save_epochs})"
            )
            plot_path = os.path.join(plot_dir, f"{model_key}.png")
            plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)
            continue

        acc4 = int(round(val_acc * 10000))
        weight_name = f"{model_key}_{acc4}.pth"
        weight_path = os.path.join(model_dir, weight_name)

        # Get all related weights
        related_weights = []
        for fname in os.listdir(model_dir):
            if fname.startswith(model_key) and fname.endswith(".pth"):
                try:
                    acc_part = fname.replace(".pth", "").split("_")[-1]
                    acc_val = float(acc_part) / 10000
                    related_weights.append((acc_val, os.path.join(model_dir, fname)))
                except Exception:
                    print(f"Skipping invalid model file: {fname}")
                    continue

        related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
        top2_accs = set(acc for acc, _ in related_weights[:2])

        if val_acc in top2_accs:
            print(
                f"⏩ Skipped saving {weight_name} (accuracy {val_acc:.6f} already in top 2)"
            )
        else:
            related_weights.append((val_acc, weight_path))
            related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
            top2_paths = set(path for _, path in related_weights[:2])

            if weight_path in top2_paths:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), weight_path)
                else:
                    torch.save(model.state_dict(), weight_path)
                print(f"✅ Saved new model: {weight_name} (acc = {val_acc:.6f})")

            for _, fname_path in related_weights:
                if fname_path not in top2_paths and os.path.exists(fname_path):
                    try:
                        os.remove(fname_path)
                        print(f"🗑️ Deleted model: {os.path.basename(fname_path)}")
                    except Exception as e:
                        print(f"⚠️ Could not delete {fname_path}: {e}")

        # Plot metrics
        plot_path = os.path.join(plot_dir, f"{model_key}.png")
        plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(),
                "finished",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )

    print(f"\nTraining finished. Best accuracy: {best_acc:.4f}")
    print(f"Training log saved to {log_file}")
    return model
