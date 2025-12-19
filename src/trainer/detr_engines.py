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
from src.utils.detr_test_utils import (
    detr_compute_iou,
    compute_classification_metrics,
    print_test_metrics,
)
from src.utils.detr_train_utils import HungarianMatcher, DETRCriterion
from src.utils.detr_metrics import compute_detr_metrics, box_iou_batch


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
                    "test_map50",
                    "test_map25",
                    "recall_iou25",
                    "lr",
                    "patience",
                    "best_acc",
                ]
            )

    train_losses, train_accs, test_losses, test_accs, train_ious, test_ious = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    best_acc, patience_counter, last_lr = 0.0, 0, lr

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_iou, num_bbox_samples = 0.0, 0

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
                            num_bbox_samples += 1

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        avg_train_iou = epoch_iou / max(num_bbox_samples, 1)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        train_ious.append(avg_train_iou)

        # Print training summary (optional, can keep or remove)
        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} "
            f"Train Acc: {epoch_acc:.4f} IoU: {avg_train_iou:.4f} Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ===== VALIDATION =====
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        all_pred_boxes, all_gt_boxes, all_obj_scores = [], [], []

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

                # Collect boxes and scores for mAP computation
                pred_obj = torch.sigmoid(outputs["obj_scores"])
                for i in range(images.size(0)):
                    if bbox_mask[i].sum() > 0:
                        all_pred_boxes.append(outputs["pred_bboxes"][i].cpu())
                        all_gt_boxes.append(bboxes[i].cpu())
                        all_obj_scores.append(pred_obj[i].cpu())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        test_losses.append(val_loss)
        test_accs.append(val_acc)

        # Compute standard DETR metrics using the new metrics module
        bbox_metrics = {}
        if len(all_pred_boxes) > 0:
            # Stack all predictions for batch processing
            pred_boxes_batch = torch.stack(all_pred_boxes)  # (N, num_queries, 4)
            gt_boxes_batch = torch.stack(all_gt_boxes)  # (N, max_gt, 4)
            obj_scores_batch = torch.stack(all_obj_scores)  # (N, num_queries)

            # Create bbox_mask for the batch
            bbox_mask_batch = torch.zeros(
                gt_boxes_batch.size(0), gt_boxes_batch.size(1)
            )
            for i, gt_box in enumerate(all_gt_boxes):
                valid_count = (gt_box.sum(dim=1) > 0).sum()
                bbox_mask_batch[i, :valid_count] = 1

            bbox_metrics = compute_detr_metrics(
                pred_boxes_batch,
                gt_boxes_batch,
                obj_scores_batch.squeeze(-1),
                bbox_mask_batch,
            )

        # Extract metrics with default values
        avg_val_iou = bbox_metrics.get("avg_iou", 0.0)
        val_map50 = bbox_metrics.get("mAP@0.5", 0.0)
        val_map25 = bbox_metrics.get("mAP@0.25", 0.0)
        recall_iou25 = bbox_metrics.get("recall@0.25", 0.0)

        test_ious.append(avg_val_iou)

        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = compute_classification_metrics(
            all_preds, all_labels, all_probs, class_names=None
        )
        metrics["accuracy"] = val_acc * 100

        print_test_metrics(metrics, avg_val_iou, val_map50, val_map25, recall_iou25)
        # Logging
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(),
                    epoch + 1,
                    round(train_losses[-1], 6),
                    round(train_accs[-1], 6),
                    round(train_ious[-1], 6),
                    round(test_losses[-1], 6),
                    round(val_acc, 6),
                    round(avg_val_iou, 6),
                    round(val_map50, 6),
                    round(val_map25, 6),
                    round(recall_iou25, 6),
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
            test_losses,
            os.path.join(plot_dir, f"{model_key}.png"),
            test_accs,
        )

    return model
