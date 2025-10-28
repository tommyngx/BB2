import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import torch
from src.data.patch_dataset import get_dataloaders
from src.data.dataloader import load_metadata
from src.trainer.engines import train_model, evaluate_model
from src.utils.common import load_config, get_arg_or_config, clear_cuda_memory
from src.trainer.train_based import parse_img_size


def prepare_data_and_model(
    data_folder,
    model,
    batch_size,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
    pretrained_model_path=None,
    img_size=None,
    target_column=None,  # thêm target_column
):
    clear_cuda_memory()
    train_df, test_df, _ = load_metadata(
        data_folder, config_path, target_column=target_column
    )
    train_loader, test_loader = get_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        num_patches=num_patches,
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
    model,
    batch_size,
    num_epochs,
    lr,
    output,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
    patience=50,
    loss_type="ce",
    model_type=None,
    img_size=None,
    pretrained_model_path=None,
    target_column=None,  # thêm target_column
):
    train_df, test_df, train_loader, test_loader, model, device = (
        prepare_data_and_model(
            data_folder,
            model,
            batch_size,
            config_path=config_path,
            num_patches=num_patches,
            arch_type=arch_type,
            pretrained_model_path=pretrained_model_path,
            img_size=img_size,
            target_column=target_column,
        )
    )
    model_name = f"{model_type}" if model_type else arch_type
    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        model_name=model_name,
        output=output,
        dataset_folder=data_folder,
        train_df=train_df,
        patience=patience,
        loss_type=loss_type,
        arch_type=arch_type,
        num_patches=num_patches,  # truyền số patch để lưu tên file đúng
    )
    return trained_model


def run_test(
    data_folder,
    model,
    batch_size,
    output,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
    pretrained_model_path=None,
    img_size=None,
    target_column=None,  # thêm target_column
):
    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        data_folder,
        model,
        batch_size,
        config_path=config_path,
        num_patches=num_patches,
        arch_type=arch_type,
        pretrained_model_path=pretrained_model_path,
        img_size=img_size,
        target_column=target_column,
    )
    print("\nEvaluation on Test Set:")
    test_loss, test_acc = evaluate_model(
        model, test_loader, device=device, mode="Test", return_loss=True
    )
    return test_loss, test_acc


def run_gradcam(
    data_folder,
    model,
    batch_size,
    output,
    config_path="config/config.yaml",
    num_patches=None,
    arch_type="patch_resnet",
    pretrained_model_path=None,
    img_size=None,
    target_column=None,
):
    import time

    train_df, test_df, _, test_loader, model, device = prepare_data_and_model(
        data_folder,
        model,
        batch_size,
        config_path=config_path,
        num_patches=num_patches,
        arch_type=arch_type,
        pretrained_model_path=pretrained_model_path,
        img_size=img_size,
        target_column=target_column,
    )
    # Determine save path based on pretrained_model_path
    if pretrained_model_path:
        gradcam_model_path = (
            pretrained_model_path.replace(".pth", "_full.pth")
            if pretrained_model_path.endswith(".pth")
            else pretrained_model_path + "_full"
        )
    else:
        gradcam_model_path = os.path.join(
            output, f"gradcam_{type(model).__name__}_full.pth"
        )
    # Use get_gradcam_layer to determine the correct gradcam layer
    model_name = type(model).__name__.lower()
    # gradcam_layer = get_gradcam_layer(model, model_name)
    gradcam_layer = get_gradcam_layer_patch(model, model_name, arch_type)
    # Default normalization (same as A.Normalize([0.5]*3, [0.5]*3))
    normalize = {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    # Calculate real inference time using a dummy input
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    model.eval()
    with torch.no_grad():
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
        inference_time = end - start  # seconds for one forward pass
    # Save dict with model, input_size, gradcam_layer, model_name, normalize, inference_time
    model_info = {
        "model": model,
        "input_size": img_size,
        "gradcam_layer": gradcam_layer,
        "model_name": type(model).__name__,
        "normalize": normalize,
        "inference_time": inference_time,
        "num_patches": num_patches,
        "arch_type": arch_type,
    }
    try:
        torch.save(model_info, gradcam_model_path)
        print(f"Saved full model and info for GradCAM to {gradcam_model_path}")
        print("Model info:")
        for k, v in model_info.items():
            if k != "model":
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"⚠️ Error saving GradCAM model: {e}")
    return gradcam_model_path


def get_gradcam_layer_patch(model, model_name, arch_type):
    """
    Return the name of the appropriate layer for GradCAM based on MIL/patch architecture.

    This function determines the optimal layer for GradCAM visualization in
    patch-based and Multi-Instance Learning models.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    model_name : str
        Lowercase name of the model class.
    arch_type : str
        Architecture type (e.g., 'mil', 'mil_v2', 'patch_resnet', 'patch_transformer').

    Returns
    -------
    gradcam_layer : str or None
        Name of the layer suitable for GradCAM. Returns None if no suitable layer found.

    Notes
    -----
    - For MIL models (mil, mil_v2-v12): targets the feature extractor's last conv layer
    - For patch_resnet: targets 'feature_extractor.layer4'
    - For patch_transformer: targets the last attention block
    - For token_mixer and global_local variants: targets specific mixer/attention layers
    - Falls back to the last layer if no specific match is found
    """
    # MIL architectures (mil, mil_v2, mil_v3, ..., mil_v12)
    if "mil" in arch_type:
        # For MIL models, check multiple possible layer names
        # Try base_model first (common in MIL v4 and newer versions)
        if hasattr(model, "base_model"):
            if hasattr(model.base_model, "layer4"):
                return "base_model.layer4"
            elif hasattr(model.base_model, "stages"):
                return "base_model.stages.3"
            elif hasattr(model.base_model, "blocks"):
                return "base_model.blocks.5"
        # # Try feature_extractor (older MIL versions)
        # elif hasattr(model, 'feature_extractor'):
        #     if hasattr(model.feature_extractor, 'layer4'):
        #         return "feature_extractor.layer4"
        #     elif hasattr(model.feature_extractor, 'stages'):
        #         return "feature_extractor.stages.3"
        #     elif hasattr(model.feature_extractor, 'blocks'):
        #         return "feature_extractor.blocks.5"
        # Fallback for MIL models
        # return "base_model.layer4" if hasattr(model, 'base_model')
        return "base_model.layer4"

    # Patch ResNet
    elif arch_type == "patch_resnet":
        # return "feature_extractor.layer4"
        return "base_model.layer4"

    # Patch Transformer
    elif arch_type == "patch_transformer":
        # Target the last transformer block
        if hasattr(model, "transformer_encoder"):
            return "transformer_encoder.layers.-1"
        return "transformer_encoder"

    # Token Mixer
    elif arch_type == "token_mixer":
        if hasattr(model, "token_mixer"):
            return "token_mixer"
        return "feature_extractor.layer4"

    # Global-Local architectures
    elif "global_local" in arch_type:
        if hasattr(model, "local_feature_extractor"):
            return "local_feature_extractor.layer4"
        elif hasattr(model, "feature_extractor"):
            return "feature_extractor.layer4"
        return "feature_extractor"

    # If not matched, try to get the feature extractor's last layer
    else:
        if hasattr(model, "feature_extractor"):
            # Try to get the last layer of feature_extractor
            fe_children = list(model.feature_extractor.named_children())
            if fe_children:
                return f"feature_extractor.{fe_children[-1][0]}"

        # Final fallback: return the last layer's name
        children = list(model.named_children())
        if children:
            return children[-1][0]
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file name in config folder",
    )
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--output", type=str)
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "gradcam"], default="train"
    )
    parser.add_argument("--patience", type=int)
    parser.add_argument("--loss_type", type=str, choices=["ce", "focal"])
    parser.add_argument("--num_patches", type=int)
    parser.add_argument(
        "--arch_type",
        type=str,
        default="patch_resnet",
        choices=[
            "patch_resnet",
            "patch_transformer",
            "token_mixer",
            "global_local",
            "global_local_token",
            "mil",
            "mil_v2",
            "mil_v3",
            "mil_v4",
            "mil_v5",
            "mil_v6",
            "mil_v7",
            "mil_v8",
            "mil_v9",
            "mil_v10",  # added v10
            "mil_v11",  # added v11
            "mil_v12",  # added v12
        ],
    )
    parser.add_argument("--img_size", type=str, default=None)
    parser.add_argument(
        "--target_column", type=str, default=None, help="Name of target column"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    data_folder = get_arg_or_config(args.data_folder, config.get("data_folder"), None)
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
    num_patches = get_arg_or_config(args.num_patches, config.get("num_patches"), 2)
    arch_type = get_arg_or_config(
        args.arch_type, config.get("arch_type"), "patch_resnet"
    )
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    target_column = get_arg_or_config(
        args.target_column, config.get("target_column"), None
    )
    if img_size is not None and isinstance(img_size, str):
        img_size = parse_img_size(img_size)

    from src.models.patch_model import get_patch_model

    train_df, test_df, class_names = load_metadata(
        data_folder, args.config, target_column=target_column, print_stats=False
    )
    num_classes = (
        len(class_names)
        if class_names is not None and len(class_names) > 0
        else len(train_df["cancer"].unique())
    )

    model = get_patch_model(
        model_type=model_type,
        num_patches=num_patches,
        arch_type=arch_type,
        num_classes=num_classes,  # assuming binary classification; adjust as needed
    )

    if args.mode == "train":
        run_train(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            output=output,
            config_path=args.config,
            num_patches=num_patches,
            arch_type=arch_type,
            patience=patience,
            loss_type=loss_type,
            model_type=model_type,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
        )
    elif args.mode == "test":
        run_test(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            output=output,
            config_path=args.config,
            num_patches=num_patches,
            arch_type=arch_type,
            pretrained_model_path=pretrained_model_path,
            img_size=img_size,
            target_column=target_column,
        )
    elif args.mode == "gradcam":
        run_gradcam(
            data_folder=data_folder,
            model=model,
            batch_size=batch_size,
            output=output,
            config_path=args.config,
            num_patches=num_patches,
            arch_type=arch_type,
            pretrained_model_path=pretrained_model_path,
            img_size=img_size,
            target_column=target_column,
        )
