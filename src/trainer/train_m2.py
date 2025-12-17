import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import random
import torch
from src.data.m2_data import get_m2_dataloaders
from src.data.dataloader import load_metadata
from src.models.m2_model import get_m2_model
from src.trainer.engines_m2 import train_m2_model, evaluate_m2_model
from src.utils.common import load_config, get_arg_or_config, clear_cuda_memory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_img_size(val):
    if val is None:
        return None
    if "x" in val:
        h, w = val.lower().split("x")
        return (int(h), int(w))
    else:
        s = int(val)
        return (s, s)


def prepare_m2_data_and_model(
    data_folder,
    model_type,
    batch_size,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    mode="train",
    target_column=None,
):
    clear_cuda_memory()
    train_df, test_df, class_names = load_metadata(
        data_folder, config_path, target_column=target_column
    )

    train_loader, test_loader = get_m2_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
        mode=mode,
    )

    model = get_m2_model(model_type=model_type, num_classes=len(class_names))
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


def sample_viz_batches(
    data_loader,
    output_dir,
    class_names,
    num_batches=5,
    seed=42,
):
    """
    Visualize a few batches as grid images with GT bbox.
    Each batch is saved as one grid image (max 16 samples per batch).
    Supports both classification and DETR dataloader.
    """
    import math

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    total_batches = len(data_loader)
    batch_indices = set(
        random.sample(range(total_batches), min(num_batches, total_batches))
    )

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx not in batch_indices:
            continue

        images = batch["image"]
        labels = batch["label"]
        # Detect DETR or classification dataloader
        is_detr = "bboxes" in batch and "bbox_mask" in batch

        batch_size = images.shape[0]
        num_samples = min(batch_size, 16)  # Max 16 samples per grid

        # Calculate grid size
        ncols = min(4, num_samples)
        nrows = math.ceil(num_samples / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        for i in range(num_samples):
            row = i // ncols
            col = i % ncols
            ax = axes[row][col]

            # Denormalize image
            img = images[i].cpu()
            img_np = (img.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

            ax.imshow(img_np)

            label = labels[i].item()
            title = f"{class_names[label]}"

            if is_detr:
                # DETR: draw all valid bboxes for this sample
                bboxes = batch["bboxes"][i].cpu().numpy()  # [max_objects, 4]
                bbox_mask = batch["bbox_mask"][i].cpu().numpy()  # [max_objects]
                h_img, w_img = img_np.shape[:2]
                num_box = 0
                for j in range(bboxes.shape[0]):
                    if bbox_mask[j] > 0.5:
                        x, y, w, h = bboxes[j]
                        x_pix = int(x * w_img)
                        y_pix = int(y * h_img)
                        w_pix = int(w * w_img)
                        h_pix = int(h * h_img)
                        if w_pix > 0 and h_pix > 0:
                            rect = mpatches.Rectangle(
                                (x_pix, y_pix),
                                w_pix,
                                h_pix,
                                linewidth=2,
                                edgecolor="lime",
                                facecolor="none",
                            )
                            ax.add_patch(rect)
                            num_box += 1
                if num_box > 0:
                    title += f"\n{num_box} bbox(es)"
            else:
                # Classification: draw single bbox if exists
                bboxes = batch["bbox"]
                has_bbox = batch["has_bbox"]
                if has_bbox[i].item() > 0 and not torch.isnan(bboxes[i]).any():
                    x, y, w, h = bboxes[i].cpu().numpy()
                    h_img, w_img = img_np.shape[:2]
                    x_pix = int(x * w_img)
                    y_pix = int(y * h_img)
                    w_pix = int(w * w_img)
                    h_pix = int(h * h_img)
                    rect = mpatches.Rectangle(
                        (x_pix, y_pix),
                        w_pix,
                        h_pix,
                        linewidth=2,
                        edgecolor="lime",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    title += f"\nBBox: {w_pix}x{h_pix}"

            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide empty subplots
        for i in range(num_samples, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row][col].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"batch_{batch_idx:03d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved batch {batch_idx} ({num_samples} samples) -> {save_path}")


def run_m2_train(
    data_folder,
    model_type,
    batch_size,
    num_epochs,
    lr,
    output,
    config_path="config/config.yaml",
    img_size=None,
    patience=50,
    loss_type="ce",
    lambda_bbox=1.0,
    pretrained_model_path=None,
    target_column=None,
    sample_viz=False,  # new param
):
    train_df, test_df, train_loader, test_loader, model, device = (
        prepare_m2_data_and_model(
            data_folder,
            model_type,
            batch_size,
            config_path=config_path,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
        )
    )

    model_name = f"{model_type}"
    class_names = (
        sorted(train_df[target_column].unique())
        if target_column
        else sorted(train_df["cancer"].unique())
    )

    # Sample visualization
    if sample_viz:
        viz_dir = os.path.join(output, "train_sample")
        print(
            f"🔍 Visualizing {min(5, len(train_loader))} random batches to {viz_dir} ..."
        )
        sample_viz_batches(train_loader, viz_dir, class_names, num_batches=5)

    trained_model = train_m2_model(
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
        arch_type="m2",
        lambda_bbox=lambda_bbox,
    )

    return trained_model


def run_m2_test(
    data_folder,
    model_type,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,
    lambda_bbox=1.0,
):
    train_df, test_df, _, test_loader, model, device = prepare_m2_data_and_model(
        data_folder,
        model_type,
        batch_size,
        config_path=config_path,
        img_size=img_size,
        pretrained_model_path=pretrained_model_path,
        mode="test",
        target_column=target_column,
    )

    print("\nEvaluation on Test Set:")
    test_loss, test_acc, test_iou = evaluate_m2_model(
        model,
        test_loader,
        device=device,
        mode="Test",
        return_loss=True,
        lambda_bbox=lambda_bbox,
    )

    if pretrained_model_path:
        full_model_path = pretrained_model_path.replace(".pth", "_full.pth")
        try:
            torch.save(model, full_model_path)
            print(f"Saved full model to {full_model_path}")
        except Exception as e:
            print(f"⚠️ Error saving full model: {e}")

    return test_loss, test_acc, test_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--output", type=str)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--patience", type=int)
    parser.add_argument(
        "--loss_type", type=str, choices=["ce", "focal", "focal2", "ldam"]
    )
    parser.add_argument("--img_size", type=str, default=None)
    parser.add_argument("--target_column", type=str, default=None)
    parser.add_argument(
        "--lambda_bbox", type=float, default=1.0, help="Weight for bbox regression loss"
    )
    parser.add_argument(
        "--sample_viz",
        action="store_true",
        help="Visualize a few training batches and their GT bbox",
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
    img_size = get_arg_or_config(args.img_size, config.get("image_size"), None)
    target_column = get_arg_or_config(
        args.target_column, config.get("target_column"), None
    )
    lambda_bbox = get_arg_or_config(args.lambda_bbox, config.get("lambda_bbox"), 1.0)

    if img_size is not None and isinstance(img_size, str):
        img_size = parse_img_size(img_size)

    if args.mode == "train":
        run_m2_train(
            data_folder=data_folder,
            model_type=model_type,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            output=output,
            config_path=args.config,
            img_size=img_size,
            patience=patience,
            loss_type=loss_type,
            lambda_bbox=lambda_bbox,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
            sample_viz=args.sample_viz,  # pass flag
        )
    elif args.mode == "test":
        run_m2_test(
            data_folder=data_folder,
            model_type=model_type,
            batch_size=batch_size,
            output=output,
            config_path=args.config,
            img_size=img_size,
            pretrained_model_path=pretrained_model_path,
            target_column=target_column,
            lambda_bbox=lambda_bbox,
        )
