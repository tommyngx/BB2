import argparse
import yaml
import os
import shutil
import warnings
import gc
import torch
from sklearn.metrics import classification_report
from data import load_data, get_dataloaders
from models import get_model
from train import train_model, evaluate_model
from visualization import plot_gradcam_plus
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import plot_confusion_matrix

torch.serialization.add_safe_globals([argparse.Namespace])
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_name):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
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


def clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def prepare_data_and_model(
    dataset_folder,
    model_type,
    batch_size,
    pretrained_model_path=None,
    num_classes=2,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
):
    clear_cuda_memory()
    train_df, test_df = load_data(dataset_folder, config_path=config_path)
    train_loader, test_loader = get_dataloaders(
        train_df,
        test_df,
        dataset_folder,
        batch_size=batch_size,
        config_path=config_path,
        num_patches=num_patches,
    )
    num_classes = train_df["cancer"].nunique()
    model = get_model(
        model_type=model_type,
        num_classes=num_classes,
        config_path=config_path,
        num_patches=num_patches,
        arch_type=arch_type,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pretrained_model_path:
        try:
            model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device)
            )
            print(f"Loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(f"⚠️ Error loading pretrained model: {e}. Training from scratch.")
    else:
        # If no pretrained path provided, try loading the best weight for the model_key
        model_dir = os.path.join("output", "models")
        dataset_name = os.path.basename(os.path.normpath(dataset_folder))
        model_key = f"{dataset_name}_{model_type}_{arch_type}_p{num_patches or 2}"
        os.makedirs(model_dir, exist_ok=True)  # Create model_dir if it doesn't exist
        weight_files = [
            f
            for f in os.listdir(model_dir)
            if f.startswith(model_key) and f.endswith(".pth")
        ]
        if weight_files:
            weight_files = sorted(
                weight_files,
                key=lambda x: float(x.replace(".pth", "").split("_")[-1]) / 10000,
                reverse=True,
            )
            weight_path = os.path.join(model_dir, weight_files[0])
            try:
                model.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"Loaded best model weight: {weight_path}")
            except Exception as e:
                print(
                    f"⚠️ Error loading best model weight {weight_path}: {e}. Using untrained model."
                )
    model = model.to(device)
    return train_df, test_df, train_loader, test_loader, model, device


def run_train(
    dataset_folder,
    model_type,
    batch_size,
    num_epochs,
    lr,
    pretrained_model_path=None,
    outputs_link=None,
    patience=50,
    loss_type="ce",
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
):
    train_df, test_df, train_loader, test_loader, model, device = (
        prepare_data_and_model(
            dataset_folder,
            model_type,
            batch_size,
            pretrained_model_path,
            config_path=config_path,
            num_patches=num_patches,
            arch_type=arch_type,
        )
    )
    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=model_type,
        output=outputs_link,
        dataset_folder=dataset_folder,
        train_df=train_df,
        patience=patience,
        loss_type=loss_type,
        config_path=config_path,
        num_patches=num_patches,
        arch_type=arch_type,
    )


def run_test(
    dataset_folder,
    model_type,
    batch_size,
    pretrained_model_path=None,
    outputs_link=None,
    gradcam=False,
    gradcam_num_images=3,
    gradcam_random_state=29,
    dataset_name=None,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
):
    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        dataset_folder,
        model_type,
        batch_size,
        pretrained_model_path,
        config_path=config_path,
        num_patches=num_patches,
        arch_type=arch_type,
    )
    model.eval()
    print("\nEvaluation on Test Set:")
    all_labels, all_preds = [], []
    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(device)
            outputs = model(patches)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
    class_names = [str(i) for i in sorted(set(all_labels))]
    if not outputs_link:
        outputs_link = "output"
    plot_dir = os.path.join(str(outputs_link), "figures")
    os.makedirs(plot_dir, exist_ok=True)
    model_key = f"{dataset_name}_{model_type}_{arch_type}_p{num_patches or 2}".replace(
        " ", ""
    )
    cm_path = os.path.join(plot_dir, f"{model_key}_confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    if gradcam:
        print(
            f"Running GradCAM++ on {gradcam_num_images} images (random_state={gradcam_random_state})..."
        )
        gradcam_dir = plot_dir
        plot_gradcam_plus(
            model,
            test_df,
            dataset_folder,
            num_images=gradcam_num_images,
            random_state=gradcam_random_state,
            save_dir=gradcam_dir,
            dataset_name=dataset_name,
            device=device,
        )


def run_gradcam(
    dataset_folder,
    model_type,
    batch_size,
    pretrained_model_path=None,
    outputs_link=None,
    gradcam_num_images=3,
    gradcam_random_state=29,
    dataset_name=None,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
):
    _, test_df, _, _, model, device = prepare_data_and_model(
        dataset_folder,
        model_type,
        batch_size,
        pretrained_model_path,
        config_path=config_path,
        num_patches=num_patches,
        arch_type=arch_type,
    )
    model.eval()
    if not outputs_link:
        outputs_link = "output"
    gradcam_dir = os.path.join(str(outputs_link), "figures")
    os.makedirs(gradcam_dir, exist_ok=True)
    print(
        f"Running GradCAM++ only on {gradcam_num_images} images (random_state={gradcam_random_state})..."
    )
    plot_gradcam_plus(
        model,
        test_df,
        dataset_folder,
        num_images=gradcam_num_images,
        random_state=gradcam_random_state,
        save_dir=gradcam_dir,
        dataset_name=dataset_name,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file name in config folder",
    )
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--outputs_link", type=str)
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "gradcam"], default="train"
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Run GradCAM++ visualization when testing",
    )
    parser.add_argument(
        "--gradcam_num_images", type=int, help="Number of images for GradCAM++"
    )
    parser.add_argument(
        "--gradcam_random_state", type=int, help="Random state for GradCAM++"
    )
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["ce", "focal"],
        help="Loss function: ce or focal",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=None,
        help="Number of patches (e.g., 2, 3, 4)",
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear output directory before training"
    )
    parser.add_argument(
        "--arch_type",
        type=str,
        choices=["patch_resnet", "patch_transformer"],
        default=None,
        help="Architecture type: patch_resnet or patch_transformer",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_folder = get_arg_or_config(
        args.dataset_folder, config.get("dataset_folder"), "your_data_folder"
    )
    model_type = get_arg_or_config(
        args.model_type, config.get("model_type"), "resnet50"
    )
    batch_size = get_arg_or_config(args.batch_size, config.get("batch_size"), 16)
    num_epochs = get_arg_or_config(args.num_epochs, config.get("num_epochs"), 10)
    lr = get_arg_or_config(args.lr, config.get("lr"), 1e-4)
    pretrained_model_path = get_arg_or_config(
        args.pretrained_model_path, config.get("pretrained_model_path"), None
    )
    outputs_link = get_arg_or_config(args.outputs_link, config.get("outputs"), "output")
    gradcam = get_arg_or_config(args.gradcam, config.get("gradcam"), False)
    gradcam_num_images = get_arg_or_config(
        args.gradcam_num_images, config.get("gradcam_num_images"), 3
    )
    gradcam_random_state = get_arg_or_config(
        args.gradcam_random_state, config.get("gradcam_random_state"), 29
    )
    patience = get_arg_or_config(args.patience, config.get("patience"), 50)
    loss_type = get_arg_or_config(args.loss_type, config.get("loss_type"), "ce")
    num_patches = get_arg_or_config(args.num_patches, config.get("num_patches"), 2)
    arch_type = get_arg_or_config(
        args.arch_type, config.get("arch_type"), "patch_resnet"
    )
    dataset_name = os.path.basename(os.path.normpath(dataset_folder))
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", args.config)

    # Clear output directory if --clear is specified
    if args.clear and os.path.exists(outputs_link):
        shutil.rmtree(outputs_link)
        print(f"Cleared output directory: {outputs_link}")
    os.makedirs(outputs_link, exist_ok=True)

    if args.mode == "train":
        run_train(
            dataset_folder=dataset_folder,
            model_type=model_type,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            pretrained_model_path=pretrained_model_path,
            outputs_link=outputs_link,
            patience=patience,
            loss_type=loss_type,
            config_path=config_path,
            num_patches=num_patches,
            arch_type=arch_type,
        )
    elif args.mode == "test":
        run_test(
            dataset_folder=dataset_folder,
            model_type=model_type,
            batch_size=batch_size,
            pretrained_model_path=pretrained_model_path,
            outputs_link=outputs_link,
            gradcam=gradcam,
            gradcam_num_images=gradcam_num_images,
            gradcam_random_state=gradcam_random_state,
            dataset_name=dataset_name,
            config_path=config_path,
            num_patches=num_patches,
            arch_type=arch_type,
        )
    elif args.mode == "gradcam":
        run_gradcam(
            dataset_folder=dataset_folder,
            model_type=model_type,
            batch_size=batch_size,
            pretrained_model_path=pretrained_model_path,
            outputs_link=outputs_link,
            gradcam_num_images=gradcam_num_images,
            gradcam_random_state=gradcam_random_state,
            dataset_name=dataset_name,
            config_path=config_path,
            num_patches=num_patches,
            arch_type=arch_type,
        )
