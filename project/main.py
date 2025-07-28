import argparse
import yaml
import os
import warnings
import gc  # thêm import gc

from data import load_data, get_dataloaders
from models import get_model
from train import train_model, evaluate_model
from visualization import plot_gradcam_plus
from utils import plot_confusion_matrix
import torch
from sklearn.metrics import classification_report

torch.serialization.add_safe_globals([argparse.Namespace])
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


def get_arg_or_config(arg_val, config_val, default_val):
    # Nếu arg_val là None hoặc không được truyền (argparse.SUPPRESS), thì lấy từ config hoặc default
    # Nếu arg_val được truyền (dù là "ce" hay "focal"), thì lấy arg_val
    # Nếu muốn bỏ qua giá trị mặc định của argparse, chỉ set default=None trong add_argument
    if arg_val is not None:
        return arg_val
    if config_val is not None:
        return config_val
    return default_val


def clear_cuda_memory():
    gc.collect()  # Thu gom rác Python
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
):
    clear_cuda_memory()  # Dọn dẹp bộ nhớ GPU trước khi bắt đầu

    train_df, test_df = load_data(dataset_folder, config_path=config_path)
    train_loader, test_loader = get_dataloaders(
        train_df,
        test_df,
        dataset_folder,
        batch_size=batch_size,
        config_path=config_path,
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
            print(f"⚠️ Error loading pretrained model: {e}. Training from scratch.")
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
    config_path="config/config.yaml",  # thêm config_path
):
    train_df, test_df, train_loader, test_loader, model, device = (
        prepare_data_and_model(
            dataset_folder,
            model_type,
            batch_size,
            pretrained_model_path,
            config_path=config_path,
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
        # pretrained_model_path=pretrained_model_path,
        output=outputs_link,
        dataset_folder=dataset_folder,
        train_df=train_df,
        patience=patience,
        loss_type=loss_type,
    )
    # print("\nEvaluation on Test Set:")
    # evaluate_model(trained_model, test_loader, device=device, mode="Test")
    # Uncomment to evaluate on train set
    # print("\nEvaluation on Train Set:")
    # evaluate_model(trained_model, train_loader, device=device, mode="Train")


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
    config_path="config/config.yaml",  # thêm config_path
):
    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        dataset_folder,
        model_type,
        batch_size,
        pretrained_model_path,
        config_path=config_path,
    )
    model.eval()
    print("\nEvaluation on Test Set:")
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
    class_names = [str(i) for i in sorted(set(all_labels))]
    if not outputs_link:
        outputs_link = "output"
    plot_dir = os.path.join(str(outputs_link), "figures")
    os.makedirs(plot_dir, exist_ok=True)
    model_key = f"{dataset_name}_{model_type}".replace(" ", "")
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
    config_path="config/config.yaml",  # thêm config_path
):
    _, test_df, _, _, model, device = prepare_data_and_model(
        dataset_folder,
        model_type,
        batch_size,
        pretrained_model_path,
        config_path=config_path,
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
    )


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
    parser.add_argument(
        "--patience", type=int, help="Early stopping patience"
    )  # thêm arg patience
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["ce", "focal"],
        # default=argparse.SUPPRESS,  # ⚠️ không gán default ở đây
        help="Loss function: ce (cross-entropy) or focal",
    )
    args = parser.parse_args()
    # print("Parsed arguments:")
    # for arg, value in vars(args).items():
    #    print(f"{arg}: {value}")

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

    dataset_name = os.path.basename(os.path.normpath(dataset_folder))

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config)

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
            config_path=config_path,  # truyền config_path
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
            config_path=config_path,  # truyền config_path
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
            config_path=config_path,  # truyền config_path
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
