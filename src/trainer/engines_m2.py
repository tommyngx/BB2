import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import csv
from datetime import datetime

from src.utils.loss import FocalLoss, LDAMLoss, FocalLoss2
from src.utils.plot import plot_metrics
from src.utils.bbox_loss import GIoULoss
from src.utils.m2_utils import compute_iou, get_lambda_bbox_schedule
from src.trainer.m2_evaluate import evaluate_m2_model


def train_m2_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10,
    lr=1e-4,
    device="cpu",
    model_name="m2_model",
    output="output",
    dataset_folder="None",
    train_df=None,
    patience=50,
    loss_type="ce",
    arch_type=None,
    lambda_bbox=1.0,
):
    """Train multi-task model"""
    # Multi-GPU support
    if device != "cpu" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Setup loss functions
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
    elif loss_type == "ldam":
        cls_criterion = LDAMLoss(
            cls_num_list=train_df["cancer"].value_counts().sort_index().tolist(),
            weight=weights,
        ).to(device)
    elif loss_type == "focal2":
        cls_criterion = FocalLoss2(alpha=weights)
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=weights)

    bbox_criterion = GIoULoss()

    # Setup optimizer and schedulers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler() if device != "cpu" else None

    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    # Setup directories and model key
    model_dir = os.path.join(output, "models")
    plot_dir = os.path.join(output, "figures")
    log_dir = os.path.join(output, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = dataset_folder.split("/")[-1]
    try:
        sample = next(iter(train_loader))
        images = sample["image"]
        if images.ndim == 4:
            img_size = (images.shape[2], images.shape[3])
        else:
            img_size = (448, 448)
    except Exception:
        img_size = (448, 448)

    imgsize_str = f"{img_size[0]}x{img_size[1]}"
    model_key = f"{dataset}_{imgsize_str}_{arch_type}_{model_name}"

    print(f"Model key: {model_key}")

    # Check for existing weights
    print(
        f"Checking for existing weights in {model_dir}\n  with model_key: {model_key}"
    )
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

    log_file = os.path.join(log_dir, f"{model_key}.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as logf:
            writer = csv.writer(logf)
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
                    "lr",
                    "patience_counter",
                    "best_acc",
                ]
            )

    lambda_freeze_epochs = 10
    lambda_warmup_epochs = 10
    lambda_full_epoch = lambda_freeze_epochs + lambda_warmup_epochs

    # Training loop
    for epoch in range(num_epochs):
        lambda_bbox_now = get_lambda_bbox_schedule(
            epoch,
            lambda_bbox_target=lambda_bbox,
            freeze_epochs=lambda_freeze_epochs,
            warmup_epochs=lambda_warmup_epochs,
        )

        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        epoch_iou = 0.0
        num_bbox_samples = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for batch in loop:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            has_bbox = batch["has_bbox"].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    # Handle both 2-output and 3-output models
                    outputs = model(images)
                    if len(outputs) == 3:
                        cls_outputs, bbox_outputs, _ = outputs
                    else:
                        cls_outputs, bbox_outputs = outputs

                    cls_loss = cls_criterion(cls_outputs, labels)

                    # Bbox loss with confident mask
                    if has_bbox.sum() > 0:
                        with torch.no_grad():
                            cls_probs = torch.softmax(cls_outputs, dim=1)
                            max_probs, _ = torch.max(cls_probs, dim=1)
                            confident_mask = (max_probs > 0.7).float()

                        pos_mask = has_bbox.bool()
                        confident_pos_mask = pos_mask & (confident_mask > 0.5)

                        if confident_pos_mask.sum() > 0:
                            bbox_loss = bbox_criterion(
                                bbox_outputs[confident_pos_mask],
                                bboxes[confident_pos_mask],
                            )
                        else:
                            bbox_loss = torch.tensor(0.0).to(device)
                    else:
                        bbox_loss = torch.tensor(0.0).to(device)

                    total_loss = cls_loss + lambda_bbox_now * bbox_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Handle both 2-output and 3-output models
                outputs = model(images)
                if len(outputs) == 3:
                    cls_outputs, bbox_outputs, _ = outputs
                else:
                    cls_outputs, bbox_outputs = outputs

                cls_loss = cls_criterion(cls_outputs, labels)

                # Bbox loss with confident mask
                if has_bbox.sum() > 0:
                    with torch.no_grad():
                        cls_probs = torch.softmax(cls_outputs, dim=1)
                        max_probs, _ = torch.max(cls_probs, dim=1)
                        confident_mask = (max_probs > 0.7).float()

                    pos_mask = has_bbox.bool()
                    confident_pos_mask = pos_mask & (confident_mask > 0.5)

                    if confident_pos_mask.sum() > 0:
                        bbox_loss = bbox_criterion(
                            bbox_outputs[confident_pos_mask], bboxes[confident_pos_mask]
                        )
                    else:
                        bbox_loss = torch.tensor(0.0).to(device)
                else:
                    bbox_loss = torch.tensor(0.0).to(device)

                total_loss = cls_loss + lambda_bbox_now * bbox_loss
                total_loss.backward()
                optimizer.step()

            running_loss += total_loss.item() * images.size(0)

            if has_bbox.sum() > 0:
                with torch.no_grad():
                    pos_mask = has_bbox.bool()
                    iou = compute_iou(bbox_outputs[pos_mask], bboxes[pos_mask])
                    epoch_iou += iou.sum().item()
                    num_bbox_samples += has_bbox.sum().item()

            _, predicted = torch.max(cls_outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=total_loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        avg_train_iou = epoch_iou / max(num_bbox_samples, 1)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        train_ious.append(avg_train_iou)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
            f"IoU: {avg_train_iou:.4f} Learning Rate: {optimizer.param_groups[0]['lr']:.6f} | Lambda: {lambda_bbox_now:.4f}"
        )

        test_loss, test_acc, test_iou = evaluate_m2_model(
            model,
            test_loader,
            device=device,
            mode="Test",
            return_loss=True,
            cls_criterion=cls_criterion,
            bbox_criterion=bbox_criterion,
            lambda_bbox=lambda_bbox_now,
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_ious.append(test_iou)

        with open(log_file, "a", newline="") as logf:
            writer = csv.writer(logf)
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    epoch + 1,
                    round(epoch_loss, 6),
                    round(epoch_acc, 6),
                    round(avg_train_iou, 6),
                    round(test_loss, 6),
                    round(test_acc, 6),
                    round(test_iou, 6),
                    optimizer.param_groups[0]["lr"],
                    patience_counter,
                    round(best_acc, 6),
                ]
            )

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(test_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < last_lr:
            print(
                f"Learning rate reduced to {current_lr:.6f}, resetting patience counter"
            )
            patience_counter = 0
            last_lr = current_lr
        else:
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if epoch < lambda_full_epoch:
            print(
                f"⏸️ Skipping model save (lambda warmup not finished, epoch {epoch + 1}/{lambda_full_epoch})"
            )
            plot_path = os.path.join(plot_dir, f"{model_key}.png")
            plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)
            continue

        acc4 = int(round(test_acc * 10000))
        weight_name = f"{model_key}_{acc4}.pth"
        weight_path = os.path.join(model_dir, weight_name)

        related_weights = []
        for fname in os.listdir(model_dir):
            if fname.startswith(model_key) and fname.endswith(".pth"):
                try:
                    acc_part = fname.replace(".pth", "").split("_")[-1]
                    acc_val = float(acc_part) / 10000
                    related_weights.append((acc_val, os.path.join(model_dir, fname)))
                except Exception:
                    continue

        related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
        top2_accs = set(acc for acc, _ in related_weights[:2])

        if test_acc not in top2_accs:
            related_weights.append((test_acc, weight_path))
            related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
            top2_paths = set(path for _, path in related_weights[:2])
            if weight_path in top2_paths:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), weight_path)
                else:
                    torch.save(model.state_dict(), weight_path)
                print(f"✅ Saved new model: {weight_name}")
            for _, fname_path in related_weights:
                if fname_path not in top2_paths and os.path.exists(fname_path):
                    try:
                        os.remove(fname_path)
                        print(f"🗑️ Deleted model: {fname_path}")
                    except Exception as e:
                        print(f"⚠️ Could not delete {fname_path}: {e}")

        plot_path = os.path.join(plot_dir, f"{model_key}.png")
        plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)

    print(f"\nTraining finished. Log saved to {log_file}")
    return model
