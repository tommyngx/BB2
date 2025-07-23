import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report
from utils import plot_metrics, plot_confusion_matrix


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
    pretrained_model_path=None,
    dataset="dataset",
    output="output",
    dataset_folder="None",
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    warmup_epochs = 5

    def warmup_lambda(epoch):
        return (
            float(epoch) / float(max(1, warmup_epochs))
            if epoch < warmup_epochs
            else 1.0
        )

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    if pretrained_model_path:
        if not os.path.exists(pretrained_model_path):
            print(
                f"Pretrained model path '{pretrained_model_path}' not found. Skipping loading."
            )
        else:
            try:
                model.load_state_dict(
                    torch.load(pretrained_model_path, map_location=device)
                )
                print(f"Loaded pretrained model from {pretrained_model_path}")
            except Exception as e:
                print(
                    f"Error loading pretrained model: {e}. Starting training from scratch."
                )

    # Prepare directories
    model_dir = os.path.join(output, "models")
    plot_dir = os.path.join(output, "plots")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    dataset = dataset_folder.split("/")[-1]

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
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

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}"
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate_model(
            model, test_loader, device=device, mode="Test", return_loss=True
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Save model with proper naming
        acc4 = int(test_acc * 10000)
        # Kiểm tra top-2 và chỉ lưu nếu hợp lệ
        model_key = f"{dataset}_{model_name}"
        acc4 = int(test_acc * 10000)
        weight_name = f"{model_key}_{acc4}.pth"
        weight_path = os.path.join(model_dir, weight_name)

        # Tìm các model liên quan cùng prefix
        related_weights = []
        for fname in os.listdir(model_dir):
            if fname.startswith(model_key) and fname.endswith(".pth"):
                try:
                    acc_part = fname.replace(".pth", "").split("_")[-1]
                    acc_val = int(acc_part) / 10000
                    related_weights.append((acc_val, os.path.join(model_dir, fname)))
                except:
                    continue

        # Thêm model hiện tại vào danh sách tạm để đánh giá top-2
        related_weights.append((test_acc, weight_path))
        related_weights = sorted(related_weights, key=lambda x: x[0], reverse=True)

        # Nếu model hiện tại nằm trong top 2 → lưu
        if (test_acc, weight_path) in related_weights[:2] and not os.path.exists(
            weight_path
        ):
            torch.save(model.state_dict(), weight_path)
            print(f"✅ Saved new best model: {weight_name} (acc = {test_acc:.4f})")

        # Xoá các model ngoài top-2
        for _, path_to_delete in related_weights[2:]:
            if os.path.exists(path_to_delete):
                try:
                    os.remove(path_to_delete)
                except Exception as e:
                    print(f"⚠️ Could not delete {path_to_delete}: {e}")

        # Save plot after every epoch
        plot_path = os.path.join(plot_dir, f"{model_key}.png")
        plot_metrics(train_losses, train_accs, test_losses, test_accs, plot_path)

        # Save confusion matrix after every epoch
        # Lấy nhãn thực và dự đoán trên test set
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
