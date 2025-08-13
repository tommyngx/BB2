import argparse
import os
import torch
from src.data.based_data import get_dataloaders
from src.data.dataloader import load_metadata
from src.trainer.engines import train_model, evaluate_model
from src.utils.common import load_config, get_arg_or_config, clear_cuda_memory


def prepare_data_and_model(
    data_folder,
    root_dir,
    model,
    batch_size,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
):
    clear_cuda_memory()
    train_df, test_df, _ = load_metadata(data_folder, config_path)
    train_loader, test_loader = get_dataloaders(
        train_df,
        test_df,
        root_dir,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
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
    model = model.to(device)
    return train_df, test_df, train_loader, test_loader, model, device


def run_train(
    data_folder,
    root_dir,
    model,
    batch_size,
    num_epochs,
    lr,
    output,
    config_path="config/config.yaml",
    img_size=None,
    patience=50,
    loss_type="ce",
):
    train_df, test_df, train_loader, test_loader, model, device = (
        prepare_data_and_model(
            data_folder,
            root_dir,
            model,
            batch_size,
            config_path=config_path,
            img_size=img_size,
        )
    )
    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=getattr(model, "name", "base_model"),
        output=output,
        dataset_folder=data_folder,
        train_df=train_df,
        patience=patience,
        loss_type=loss_type,
    )
    return trained_model


def run_test(
    data_folder,
    root_dir,
    model,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
):
    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        data_folder,
        root_dir,
        model,
        batch_size,
        config_path=config_path,
        img_size=img_size,
        pretrained_model_path=pretrained_model_path,
    )
    print("\nEvaluation on Test Set:")
    # Sử dụng evaluate_model từ engines để in report và trả về loss/acc
    test_loss, test_acc = evaluate_model(
        model, test_loader, device=device, mode="Test", return_loss=True
    )
    return test_loss, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file name in config folder",
    )
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--patience", type=int)
    parser.add_argument("--loss_type", type=str, choices=["ce", "focal"])
    parser.add_argument(
        "--img_size", type=str, default=None, help="Image size, e.g. 448 or 448x448"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    def parse_img_size(val):
        if val is None:
            return None
        if "x" in val:
            h, w = val.lower().split("x")
            return (int(h), int(w))
        else:
            s = int(val)
            return (s, s)

    # Sử dụng get_arg_or_config cho tất cả các tham số
    data_folder = get_arg_or_config(args.data_folder, config.get("data_folder"), None)
    root_dir = get_arg_or_config(args.root_dir, config.get("root_dir"), None)
    model_type = get_arg_or_config(args.model_type, config.get("model_type"), None)
    batch_size = get_arg_or_config(args.batch_size, config.get("batch_size"), 16)
    num_epochs = get_arg_or_config(args.num_epochs, config.get("num_epochs"), 10)
    lr = get_arg_or_config(args.lr, config.get("lr"), 1e-4)
    pretrained_model_path = get_arg_or_config(
        args.pretrained_model_path, config.get("pretrained_model_path"), None
    )
    output = get_arg_or_config(args.output, config.get("output"), "output")
    patience = get_arg_or_config(args.patience, config.get("patience"), 50)
    loss_type = get_arg_or_config(args.loss_type, config.get("loss_type"), "ce")
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    if img_size is not None and isinstance(img_size, str):
        img_size = parse_img_size(img_size)

    # Model creation (replace with your model factory)
    from src.models.based_model import get_based_model

    model = get_based_model(model_type=model_type)

    if args.mode == "train":
        run_train(
            data_folder=data_folder,
            root_dir=root_dir,
            model=model,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            output=output,
            config_path=args.config,
            img_size=img_size,
            patience=patience,
            loss_type=loss_type,
        )
    elif args.mode == "test":
        run_test(
            data_folder=data_folder,
            root_dir=root_dir,
            model=model,
            batch_size=batch_size,
            output=output,
            config_path=args.config,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,
        )
