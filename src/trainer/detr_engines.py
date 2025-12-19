"""
Training engine for DETR model
- Hungarian matching
- DETR loss computation
- Training loop with metrics
"""

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
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

from src.utils.loss import FocalLoss, LDAMLoss, FocalLoss2
from src.utils.plot import plot_metrics
from src.utils.bbox_loss import GIoULoss
from src.utils.detr_common_utils import box_iou
from src.utils.detr_test_utils import detr_compute_iou


class HungarianMatcher(nn.Module):
    """Hungarian matching for DETR"""

    def __init__(self, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_bboxes, target_bboxes, target_mask):
        B, N = pred_bboxes.shape[:2]
        indices = []

        for i in range(B):
            valid_mask = target_mask[i] > 0.5
            num_valid = valid_mask.sum().item()

            if num_valid == 0:
                indices.append(([], []))
                continue

            pred = pred_bboxes[i]
            tgt = target_bboxes[i][valid_mask]

            try:
                cost_bbox = torch.cdist(pred, tgt, p=1)
                cost_giou = -box_iou(pred, tgt)
                C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
                pred_idx, tgt_idx = linear_sum_assignment(C.cpu().numpy())

                if len(pred_idx) > 0:
                    assert max(pred_idx) < N and max(tgt_idx) < num_valid

                indices.append((pred_idx.tolist(), tgt_idx.tolist()))
            except Exception as e:
                print(f"⚠️ Hungarian error: {e}")
                indices.append(([], []))

        return indices


class DETRCriterion(nn.Module):
    """DETR loss with auxiliary losses"""

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
        cls_loss = self.cls_criterion(outputs["cls_logits"], targets["label"])

        indices = self.matcher(
            outputs["pred_bboxes"], targets["bboxes"], targets["bbox_mask"]
        )

        bbox_loss, giou_loss, obj_loss = self._compute_detection_losses(
            outputs["pred_bboxes"],
            outputs["obj_scores"],
            targets["bboxes"],
            targets["bbox_mask"],
            indices,
        )

        total_loss = (
            cls_loss
            + self.lambda_bbox * bbox_loss
            + self.lambda_giou * giou_loss
            + self.lambda_obj * obj_loss
        )

        aux_loss = torch.tensor(0.0, device=total_loss.device)
        if "aux_outputs" in outputs and len(outputs["aux_outputs"]) > 0:
            for aux_out in outputs["aux_outputs"]:
                aux_bbox, aux_giou, aux_obj = self._compute_detection_losses(
                    aux_out["pred_bboxes"],
                    aux_out["obj_scores"],
                    targets["bboxes"],
                    targets["bbox_mask"],
                    indices,
                )
                aux_loss += (
                    self.lambda_bbox * aux_bbox
                    + self.lambda_giou * aux_giou
                    + self.lambda_obj * aux_obj
                )

            aux_loss = aux_loss * self.aux_loss_weight / len(outputs["aux_outputs"])
            total_loss += aux_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "giou_loss": giou_loss,
            "obj_loss": obj_loss,
            "aux_loss": aux_loss,
        }

    def _compute_detection_losses(
        self, pred_bboxes, pred_obj, target_bboxes, target_mask, indices
    ):
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

            valid_mask = target_mask[i] > 0.5
            if valid_mask.sum() == 0:
                obj_loss += nn.functional.binary_cross_entropy_with_logits(
                    pred_obj[i], obj_target, reduction="sum"
                )
                continue

            valid_targets = target_bboxes[i][valid_mask]

            try:
                matched_pred = pred_bboxes[i, pred_idx]
                matched_tgt = valid_targets[tgt_idx]

                if (matched_pred < 0).any() or (matched_pred > 1).any():
                    raise ValueError("Invalid pred bbox")
                if (matched_tgt < 0).any() or (matched_tgt > 1).any():
                    raise ValueError("Invalid target bbox")

                bbox_loss += self.bbox_l1_loss(matched_pred, matched_tgt).sum()
                giou_loss += self.giou_loss(matched_pred, matched_tgt)
                obj_target[pred_idx] = 1.0
                num_boxes += len(pred_idx)
            except Exception as e:
                print(f"⚠️ Loss error: {e}")

            obj_loss += nn.functional.binary_cross_entropy_with_logits(
                pred_obj[i], obj_target, reduction="sum"
            )

        if num_boxes > 0:
            bbox_loss = bbox_loss / num_boxes
            giou_loss = giou_loss / num_boxes
        obj_loss = obj_loss / (B * N)

        return bbox_loss, giou_loss, obj_loss


def train_detr_model(
    model,
    train_loader,
    test_loader,
    num_epochs=100,
    lr=1e-4,
    device="cpu",
    model_name="detr",
    output="output",
    dataset_folder="None",
    train_df=None,
    patience=50,
    loss_type="ce",
    lambda_bbox=5.0,
    lambda_giou=2.0,
    lambda_obj=1.0,
):
    """Train DETR model"""
    model = model.to(device)

    # Classification loss
    if train_df is not None:
        class_counts = train_df["cancer"].value_counts()
        weights = torch.tensor(
            [
                len(train_df) / (len(class_counts) * class_counts[i])
                for i in range(len(class_counts))
            ],
            dtype=torch.float,
        ).to(device)
    else:
        weights = None

    if loss_type == "focal":
        cls_criterion = FocalLoss(alpha=weights)
    elif loss_type == "ldam":
        cls_criterion = LDAMLoss(
            cls_num_list=train_df["cancer"].value_counts().sort_index().tolist(),
            weight=weights,
        ).to(device)
    elif loss_type == "focal2":
        cls_criterion = FocalLoss2(alpha=weights)
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=weights)

    criterion = DETRCriterion(cls_criterion, lambda_bbox, lambda_giou, lambda_obj, 0.5)

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

    warmup_epochs = 5
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: (e + 1) / warmup_epochs
        if e < warmup_epochs
        else 0.95 ** (e - warmup_epochs),
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    # Setup directories
    model_dir = os.path.join(output, "models")
    plot_dir = os.path.join(output, "figures")
    log_dir = os.path.join(output, "logs")
    for d in [model_dir, plot_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    dataset = dataset_folder.split("/")[-1]
    try:
        sample = next(iter(train_loader))
        img_size = (sample["image"].shape[2], sample["image"].shape[3])
    except:
        img_size = (448, 448)

    model_key = f"{dataset}_{img_size[0]}x{img_size[1]}_detr_{model_name}"
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

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    best_acc, patience_counter, last_lr = 0.0, 0, lr

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_iou, num_bbox = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bboxes"].to(device)
            bbox_mask = batch["bbox_mask"].to(device)

            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(images)
                    loss_dict = criterion(
                        outputs,
                        {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask},
                    )
                    loss = loss_dict["total_loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss_dict = criterion(
                    outputs, {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask}
                )
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs["cls_logits"], 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if bbox_mask.sum() > 0:
                with torch.no_grad():
                    for i in range(images.size(0)):
                        if bbox_mask[i].sum() > 0:
                            pred = outputs["pred_bboxes"][i, 0:1]
                            tgt = bboxes[i][bbox_mask[i] > 0.5][:1]
                            epoch_iou += detr_compute_iou(pred[0], tgt[0]).item()
                            num_bbox += 1

        train_losses.append(running_loss / total)
        train_accs.append(correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_iou, num_val_bbox = 0.0, 0
        all_preds, all_labels, all_probs = [], [], []
        all_pred_boxes, all_gt_boxes = [], []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bboxes"].to(device)
                bbox_mask = batch["bbox_mask"].to(device)

                outputs = model(images)
                loss_dict = criterion(
                    outputs, {"label": labels, "bboxes": bboxes, "bbox_mask": bbox_mask}
                )

                val_loss += loss_dict["total_loss"].item() * images.size(0)
                probs = torch.softmax(outputs["cls_logits"], dim=1)
                _, predicted = torch.max(outputs["cls_logits"], 1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                pred_obj = torch.sigmoid(outputs["obj_scores"])
                for i in range(images.size(0)):
                    if bbox_mask[i].sum() > 0:
                        top_idx = pred_obj[i].squeeze(-1).argmax()
                        pred_box = outputs["pred_bboxes"][i, top_idx : top_idx + 1]
                        gt_boxes = bboxes[i][bbox_mask[i] > 0.5]

                        val_iou += detr_compute_iou(pred_box[0], gt_boxes[0]).item()
                        num_val_bbox += 1
                        all_pred_boxes.append(pred_box.cpu())
                        all_gt_boxes.append(gt_boxes.cpu())

        val_acc = val_correct / val_total
        test_losses.append(val_loss / val_total)
        test_accs.append(val_acc)

        val_map = sum(
            1
            for p, g in zip(all_pred_boxes, all_gt_boxes)
            if detr_compute_iou(p[0], g[0]).item() >= 0.5
        ) / max(len(all_pred_boxes), 1)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        try:
            auc = (
                roc_auc_score(all_labels, np.array(all_probs)[:, 1]) * 100
                if np.array(all_probs).shape[1] == 2
                else 0.0
            )
        except:
            auc = 0.0

        print(
            f"\nEpoch {epoch + 1}: Train Loss={train_losses[-1]:.4f} Acc={train_accs[-1]:.4f}"
        )
        print(f"Test Acc={val_acc * 100:.2f}% AUC={auc:.2f}% mAP={val_map * 100:.2f}%")

        # Logging
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(),
                    epoch + 1,
                    round(train_losses[-1], 6),
                    round(train_accs[-1], 6),
                    round(epoch_iou / max(num_bbox, 1), 6),
                    round(test_losses[-1], 6),
                    round(val_acc, 6),
                    round(val_iou / max(num_val_bbox, 1), 6),
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
            plateau_scheduler.step(val_loss / val_total)

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < last_lr:
            patience_counter, last_lr = 0, current_lr
        else:
            if val_acc > best_acc:
                best_acc, patience_counter = val_acc, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save top-2 models
        if epoch >= 10:
            acc4 = int(round(val_acc * 10000))
            weight_path = os.path.join(model_dir, f"{model_key}_{acc4}.pth")

            related = [
                (
                    float(f.split("_")[-1].replace(".pth", "")) / 10000,
                    os.path.join(model_dir, f),
                )
                for f in os.listdir(model_dir)
                if f.startswith(model_key) and f.endswith(".pth")
            ]
            related.append((val_acc, weight_path))
            related = sorted(related, reverse=True)
            top2 = set(p for _, p in related[:2])

            if weight_path in top2:
                torch.save(
                    model.state_dict()
                    if not isinstance(model, nn.DataParallel)
                    else model.module.state_dict(),
                    weight_path,
                )
                print(f"✅ Saved: {os.path.basename(weight_path)}")

            for _, p in related:
                if p not in top2 and os.path.exists(p):
                    os.remove(p)

        plot_metrics(
            train_losses,
            train_accs,
            test_losses,
            test_accs,
            os.path.join(plot_dir, f"{model_key}.png"),
        )

    return model
