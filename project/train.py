import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report
from utils import plot_metrics, plot_confusion_matrix

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Add AMP import
import torch.cuda.amp
import copy


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # alpha có thể là None, scalar, hoặc tensor [w_0, w_1, ...]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Tính cross-entropy loss (không áp weight trực tiếp)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Tính xác suất lớp đúng (pt)
        pt = torch.exp(-ce_loss)

        # Tính focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Áp dụng alpha nếu có
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Nếu alpha là scalar, nhân trực tiếp
                focal_loss = self.alpha * focal_loss
            else:
                # Nếu alpha là tensor trọng số lớp
                alpha_t = self.alpha[targets]  # Lấy alpha tương ứng với lớp
                focal_loss = alpha_t * focal_loss

        # Áp dụng reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def evaluate_model(model, data_loader, device="cpu", mode="Test", return_loss=False):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    acc = correct / total
    avg_loss = total_loss / total
    print(f"{mode} Accuracy: {acc:.4f} | Loss: {avg_loss:.4f}")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    return (avg_loss, acc) if return_loss else acc


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10,
    lr=1e-4,
    device="cpu",
    model_name="model",
    dataset="dataset",
    output="output",
    dataset_folder="None",
    train_df=None,
    patience=50,
    loss_type="ce",
    use_ema=True,  # <--- add argument to enable EMA
    ema_decay=0.999,  # <--- EMA decay
    soft_positive_label=True,  # <--- enable soft positive label
):
    # Không cần thay đổi gì ở đây, vì train_df["cancer"] đã là nhãn số liên tục
    model = model.to(device)
    # Tính trọng số cho loss
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

    # Chọn loss function
    if loss_type == "focal":
        criterion = FocalLoss(alpha=weights)
        print("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(
            "Using CrossEntropyLoss (with weight)"
            if weights is not None
            else "Using CrossEntropyLoss"
        )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # AMP: create scaler if using CUDA
    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Warm-up scheduler: linearly increase lr for first 5 epochs
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

    # Prepare directories
    model_dir = os.path.join(output, "models")
    plot_dir = os.path.join(output, "figures")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    dataset = dataset_folder.split("/")[-1]
    # Thêm imgsize vào model_key (lấy trực tiếp từ batch shape)
    img_size = None
    # Lấy kích thước ảnh từ một batch trong train_loader
    try:
        sample = next(iter(train_loader))
        images, _ = sample
        if images.ndim == 4:
            img_size = (images.shape[2], images.shape[3])  # (H, W)
    except Exception:
        img_size = (448, 448)
    imgsize_str = f"{img_size[0]}x{img_size[1]}"
    model_key = f"{dataset}_{model_name}_{imgsize_str}"

    # Print existing weights in model_dir
    print(f"Checking for existing weights in {model_dir} with model_key: {model_key}")
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

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    best_acc = 0.0
    patience_counter = 0
    last_lr = lr

    # Model EMA setup
    if use_ema:
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad_(False)
        print("Model EMA enabled")
    else:
        ema_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # --- Soft positive label ---
            if (
                soft_positive_label and model.module.num_classes == 2
                if hasattr(model, "module")
                else getattr(model, "num_classes", 2) == 2
            ):
                # Only for binary classification
                targets = labels.float()
                soft_targets = torch.where(
                    targets == 1,
                    torch.full_like(targets, 0.8),
                    torch.zeros_like(targets),
                )
                outputs = model(images)
                if loss_type == "focal":
                    loss = criterion(outputs, labels)
                    # FocalLoss expects integer targets, so skip soft label for focal
                    loss = criterion(outputs, labels)
                else:
                    loss_fn = (
                        nn.BCEWithLogitsLoss(weight=weights)
                        if weights is not None
                        else nn.BCEWithLogitsLoss()
                    )
                    # outputs: [B, 1] or [B], soft_targets: [B]
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(1)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            loss = loss_fn(outputs, soft_targets)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if loss_type == "focal":
                            loss = criterion(outputs, labels)
                        else:
                            loss = loss_fn(outputs, soft_targets)
                        loss.backward()
                        optimizer.step()
                # For metrics
                preds = (torch.sigmoid(outputs) > 0.5).long()
                # Ensure preds and labels are [B]
                preds = preds.view(-1)
                labels_flat = labels.view(-1)
                running_loss += loss.item() * images.size(0)
                correct += (preds == labels_flat).sum().item()
                total += labels_flat.size(0)
                loop.set_postfix(loss=loss.item())
                # EMA update
                if use_ema:
                    with torch.no_grad():
                        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                            ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                continue
            # --- End soft positive label ---

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

            # EMA update
            if use_ema:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Evaluate on test set (use EMA model if enabled)
        eval_model = ema_model if use_ema else model
        test_loss, test_acc = evaluate_model(
            eval_model, test_loader, device=device, mode="Test", return_loss=True
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Update scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(test_loss)

        # Check if learning rate was reduced
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < last_lr:
            print(
                f"Learning rate reduced to {current_lr:.6f}, resetting patience counter"
            )
            patience_counter = 0
            last_lr = current_lr
        else:
            # Early stopping
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save model with proper naming (keep acc4 naming)
        acc4 = int(
            round(test_acc * 10000)
        )  # Sử dụng round thay vì chỉ int để làm tròn đúng
        weight_name = f"{model_key}_{acc4}.pth"
        weight_path = os.path.join(model_dir, weight_name)

        # Refresh related_weights by scanning model_dir
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

        # Add current epoch's model with full test_acc for precise sorting
        # related_weights.append((test_acc, weight_path))

        # Sort by accuracy (using full float precision), descending, and keep top-2
        related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
        top2_accs = set(acc for acc, _ in related_weights[:2])

        if test_acc in top2_accs:
            print(
                f"⏩ Skipped saving {weight_name} (accuracy {test_acc:.6f} already in top 2)"
            )
        else:
            related_weights.append((test_acc, weight_path))
            related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
            top2_paths = set(path for _, path in related_weights[:2])
            if weight_path in top2_paths:
                if use_ema:
                    torch.save(ema_model.state_dict(), weight_path)
                else:
                    torch.save(model.state_dict(), weight_path)
                print(f"✅ Saved new model: {weight_name} (acc = {test_acc:.6f})")
            # Remove all models outside top-2
            for _, fname_path in related_weights:
                if fname_path not in top2_paths and os.path.exists(fname_path):
                    try:
                        os.remove(fname_path)
                        print(f"🗑️ Deleted model: {fname_path}")
                    except Exception as e:
                        print(f"⚠️ Could not delete {fname_path}: {e}")

        # Save plot after every epoch
        plot_path = os.path.join(plot_dir, f"{model_key}.png")
        plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)

        # Save confusion matrix after every epoch (use EMA model if enabled)
        all_labels = []
        all_preds = []
        eval_model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = eval_model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        cm_path = os.path.join(plot_dir, f"{model_key}_confusion_matrix.png")
        class_names = [str(i) for i in sorted(set(all_labels))]
        # plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)

    print(f"{model_name} training finished.")
    return ema_model if use_ema else model
