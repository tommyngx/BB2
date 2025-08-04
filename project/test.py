import argparse
import yaml
import os
import torch
import warnings
from data import load_data, get_dataloaders
from models import get_model
import numpy as np
from sklearn.metrics import classification_report
from utils import plot_cm_roc_multiclass

torch.serialization.add_safe_globals([argparse.Namespace])
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_name):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    config_path = os.path.join(config_dir, config_name)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        return config["config"]
    return config


def get_arg_or_config(arg_val, config_val, default_val):
    if arg_val is not None:
        return arg_val
    if config_val is not None:
        return config_val
    return default_val


def parse_img_size(val):
    if val is None:
        return None
    if "x" in str(val):
        h, w = str(val).lower().split("x")
        return (int(h), int(w))
    else:
        s = int(val)
        return (s, s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--img_size", type=str, help="Image size, e.g. 448 or 448x448")
    parser.add_argument(
        "--outputs_link", type=str, help="Output directory to save report/results"
    )
    args = parser.parse_args()

    # Always load config from config/config.yaml (relative to project root)
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    config = load_config(config_path)
    dataset_folder = get_arg_or_config(
        args.dataset_folder, config.get("dataset_folder"), "your_data_folder"
    )
    model_type = get_arg_or_config(
        args.model_type, config.get("model_type"), "resnet50"
    )
    batch_size = get_arg_or_config(args.batch_size, config.get("batch_size"), 16)
    pretrained_model_path = get_arg_or_config(
        args.pretrained_model_path, config.get("pretrained_model_path"), None
    )
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    outputs_link = get_arg_or_config(
        args.outputs_link, config.get("outputs"), "figures"
    )
    if img_size is not None:
        if isinstance(img_size, str):
            img_size = parse_img_size(img_size)
    else:
        img_size = (224, 224)

    train_df, test_df = load_data(dataset_folder, config_path=config_path)
    # Gọi get_dataloaders mà không truyền num_workers, pin_memory (vì hàm không nhận các tham số này)
    _, test_loader = get_dataloaders(
        train_df,
        test_df,
        dataset_folder,
        batch_size=batch_size,
        config_path=config_path,
        num_workers=0,  # Thử với 8 workers
        pin_memory=False,
    )
    num_classes = train_df["cancer"].nunique()
    model = get_model(model_type=model_type, num_classes=num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrained_model_path:
        try:
            model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device)
            )
            print(f"Loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(
                f"⚠️ Error loading pretrained model: {e}. Testing with randomly initialized model."
            )
    model = model.to(device)
    model.eval()

    all_labels, all_preds = [], []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # shape [batch, n_classes]

    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    print(report)
    if outputs_link:
        os.makedirs(outputs_link, exist_ok=True)
        class_names = [str(i) for i in sorted(set(all_labels))]
        cmroc_path = os.path.join(outputs_link, "cm_roc.png")
        plot_cm_roc_multiclass(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            class_names=class_names,
            title=f"{model_type} - Confusion Matrix & ROC",
        )


if __name__ == "__main__":
    main()
