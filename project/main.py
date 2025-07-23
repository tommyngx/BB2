import argparse
import yaml
import os
from data import load_data, get_dataloaders
from models import get_model
from train import train_model, evaluate_model
from visualization import plot_gradcam_plus
from utils import plot_confusion_matrix

# Nếu bạn muốn ẩn các cảnh báo FutureWarning từ timm khi chạy script, hãy thêm đoạn sau vào đầu file main.py:
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_name):
    # Nếu chỉ truyền tên file, tự động tìm trong thư mục config
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


def run_model(
    dataset_folder,
    model_type="resnet50",
    batch_size=16,
    num_epochs=10,
    lr=1e-4,
    pretrained_model_path=None,
    outputs=None,
):
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    train_df, test_df = load_data(dataset_folder)
    train_loader, test_loader = get_dataloaders(
        train_df, test_df, dataset_folder, batch_size=batch_size
    )
    model = get_model(model_type=model_type, num_classes=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=model_type,
        pretrained_model_path=pretrained_model_path,
        output=outputs,
        dataset_folder=dataset_folder,
    )

    print("\nEvaluation on Test Set:")
    evaluate_model(trained_model, test_loader, device=device, mode="Test")
    print("\nEvaluation on Train Set:")
    evaluate_model(trained_model, train_loader, device=device, mode="Train")


def train_only(
    dataset_folder,
    model_type="resnet50",
    batch_size=16,
    num_epochs=10,
    lr=1e-4,
    pretrained_model_path=None,
    outputs_link=None,
):
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    train_df, test_df = load_data(dataset_folder)
    train_loader, test_loader = get_dataloaders(
        train_df, test_df, dataset_folder, batch_size=batch_size
    )
    model = get_model(model_type=model_type, num_classes=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=model_type,
        pretrained_model_path=pretrained_model_path,
        output=outputs_link,
        dataset_folder=dataset_folder,
    )

    print("\nEvaluation on Test Set:")
    evaluate_model(trained_model, test_loader, device=device, mode="Test")
    # print("\nEvaluation on Train Set:")
    # evaluate_model(trained_model, train_loader, device=device, mode="Train")


def test_only(
    dataset_folder,
    model_type="resnet50",
    batch_size=16,
    pretrained_model_path=None,
    outputs_link=None,
    gradcam=False,
    gradcam_num_images=None,
    gradcam_random_state=None,
    dataset_name=None,
):
    import torch
    import os

    # Load data
    train_df, test_df = load_data(dataset_folder)
    train_loader, test_loader = get_dataloaders(
        train_df, test_df, dataset_folder, batch_size=batch_size
    )
    model = get_model(model_type=model_type, num_classes=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if not outputs or not isinstance(outputs, str):
    #    outputs = "output"
    # plot_dir = os.path.join(outputs, "plots")
    # print(f"Plotting confusion matrix to {plot_dir}")

    # Load pretrained weights
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"Loaded pretrained model from {pretrained_model_path}")

    model = model.to(device)
    model.eval()

    # Evaluate on test set
    print("\nEvaluation on Test Set:")
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    from sklearn.metrics import classification_report

    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    # Plot confusion matrix
    class_names = [str(i) for i in sorted(set(all_labels))]
    # Fix: avoid ambiguous boolean check for torch.Tensor
    # if isinstance(outputs, torch.Tensor):
    #    outputs = outputs.item() if outputs.numel() == 1 else "output"
    if outputs_link is None or outputs_link == "":
        outputs_link = "output"
    plot_dir = os.path.join(str(outputs_link), "figures")
    # print(f"Plotting confusion matrix to {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)
    cm_path = os.path.join(plot_dir, f"{model_type}_confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # GradCAM++ visualization
    if gradcam:
        from visualization import plot_gradcam_plus

        num_images = gradcam_num_images if gradcam_num_images is not None else 3
        random_state = gradcam_random_state if gradcam_random_state is not None else 29
        print(
            f"Running GradCAM++ on {num_images} images (random_state={random_state})..."
        )
        # Ensure outputs_link is valid
        if outputs_link is None or outputs_link == "":
            outputs_link = "output"
        gradcam_dir = os.path.join(str(outputs_link), "figures")
        os.makedirs(gradcam_dir, exist_ok=True)
        plot_gradcam_plus(
            model,
            test_df,
            dataset_folder,
            num_images=num_images,
            random_state=random_state,
            save_dir=gradcam_dir,
            dataset_name=dataset_name,
        )


def gradcam_only(
    dataset_folder,
    model_type="resnet50",
    batch_size=16,
    pretrained_model_path=None,
    outputs_link=None,
    gradcam_num_images=None,
    gradcam_random_state=None,
    dataset_name=None,
):
    import torch
    import os

    train_df, test_df = load_data(dataset_folder)
    model = get_model(model_type=model_type, num_classes=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"Loaded pretrained model from {pretrained_model_path}")

    model = model.to(device)
    model.eval()

    from visualization import plot_gradcam_plus

    num_images = gradcam_num_images if gradcam_num_images is not None else 3
    random_state = gradcam_random_state if gradcam_random_state is not None else 29
    print(
        f"Running GradCAM++ only on {num_images} images (random_state={random_state})..."
    )

    # Ensure outputs_link is valid
    if outputs_link is None or outputs_link == "":
        outputs_link = "output"
    gradcam_dir = os.path.join(str(outputs_link), "figures")
    os.makedirs(gradcam_dir, exist_ok=True)

    # Truyền dataset_name nếu có
    plot_gradcam_plus(
        model,
        test_df,
        dataset_folder,
        num_images=num_images,
        random_state=random_state,
        save_dir=gradcam_dir,
        dataset_name=dataset_name,
    )


def get_arg_or_config(arg_val, config_val, default_val):
    # Trả về arg nếu hợp lệ, nếu không lấy từ config, nếu không lấy default
    if arg_val is not None:
        return arg_val
    if config_val is not None:
        return config_val
    return default_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file name in config folder (e.g. config.yaml)",
    )
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--outputs_link", type=str)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "gradcam"],
        default="train",
        help="train, test or gradcam only",
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
    args = parser.parse_args()

    # print("[INFO] Arguments from argparse:")
    # for k, v in vars(args).items():
    #    print(f"  {k}: {v}")

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

    dataset_name = os.path.basename(os.path.normpath(dataset_folder))

    if args.mode == "train":
        train_only(
            dataset_folder=dataset_folder,
            model_type=model_type,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            pretrained_model_path=pretrained_model_path,
            outputs_link=outputs_link,
        )
    elif args.mode == "test":
        test_only(
            dataset_folder=dataset_folder,
            model_type=model_type,
            batch_size=batch_size,
            pretrained_model_path=pretrained_model_path,
            outputs_link=outputs_link,
            gradcam=gradcam,
            gradcam_num_images=gradcam_num_images,
            gradcam_random_state=gradcam_random_state,
            dataset_name=dataset_name,
        )
    elif args.mode == "gradcam":
        gradcam_only(
            dataset_folder=dataset_folder,
            model_type=model_type,
            batch_size=batch_size,
            pretrained_model_path=pretrained_model_path,
            outputs_link=outputs_link,
            gradcam_num_images=gradcam_num_images,
            gradcam_random_state=gradcam_random_state,
            dataset_name=dataset_name,
        )

    # Ví dụ chạy train:
    # python main.py --mode train --config config.yaml --dataset_folder /path/to/data --model_type resnet50 --outputs output

    # Ví dụ chạy test:
    # python main.py --mode test --config config.yaml --dataset_folder /path/to/data --model_type resnet50 --pretrained_model_path output/models/your_dataset_resnet50_XXXX.pth --outputs output

    # Ví dụ chạy test với GradCAM++:
    # python main.py --mode test --gradcam --gradcam_num_images 5 --gradcam_random_state 42 --config config.yaml --dataset_folder /path/to/data --model_type resnet50 --pretrained_model_path output/models/your_dataset_resnet50_XXXX.pth --outputs output
    # Ví dụ chạy test:
    # python main.py --mode test --config config.yaml --dataset_folder /path/to/data --model_type resnet50 --pretrained_model_path output/models/your_dataset_resnet50_XXXX.pth --outputs output

    # Ví dụ chạy test với GradCAM++:
    # python main.py --mode test --gradcam --gradcam_num_images 5 --gradcam_random_state 42 --config config.yaml --dataset_folder /path/to/data --model_type resnet50 --pretrained_model_path output/models/your_dataset_resnet50_XXXX.pth --outputs output
