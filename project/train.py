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
    model_key = f"{dataset}_{model_name}"

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

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            # In phân phối class trong batch (tùy chọn, có thể bình luận nếu không cần)
            # print(f"Batch class distribution: {torch.bincount(labels)}")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate_model(
            model, test_loader, device=device, mode="Test", return_loss=True
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
        related_weights.append((test_acc, weight_path))

        # Sort by accuracy (using full float precision), descending, and keep top-2
        related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)
        top2 = related_weights[:2]
        top2_paths = set([path for _, path in top2])

        # Save current model if it's in top-2
        if weight_path in top2_paths:
            if os.path.exists(weight_path):
                existing_acc = (
                    float(weight_path.split("_")[-1].replace(".pth", "")) / 10000
                )
                if test_acc > existing_acc:
                    torch.save(model.state_dict(), weight_path)
                    print(f"✅ Overwrote model: {weight_name} (acc = {test_acc:.6f})")
            else:
                torch.save(model.state_dict(), weight_path)
                print(f"✅ Saved new best model: {weight_name} (acc = {test_acc:.6f})")

        # Remove all models outside top-2
        for _, path_to_delete in related_weights[2:]:
            if os.path.exists(path_to_delete) and path_to_delete not in top2_paths:
                try:
                    os.remove(path_to_delete)
                    print(f"🗑️ Deleted model: {path_to_delete}")
                except Exception as e:
                    print(f"⚠️ Could not delete {path_to_delete}: {e}")

        # Save plot after every epoch
        plot_path = os.path.join(plot_dir, f"{model_key}.png")
        plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)

        # Save confusion matrix after every epoch
        all_labels = []
        all_preds = []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        cm_path = os.path.join(plot_dir, f"{model_key}_confusion_matrix.png")
        class_names = [str(i) for i in sorted(set(all_labels))]
        plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)

    print(f"{model_name} training finished.")
    return model
