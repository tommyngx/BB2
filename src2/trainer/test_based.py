"""
Testing script for classification model with GradCAM visualization
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from src2.data.based_data import get_dataloaders
from src2.data.dataloader import load_metadata
from src2.models.based_model import get_based_model
from src2.utils.common import load_config, get_arg_or_config
from src2.utils.gradcam_utils import get_gradcam_layer, gradcam
from src2.utils.based_viz_utils import visualize_classification_result
from src2.utils.based_test_utils import (
    evaluate_classification_model,
    compute_classification_metrics,
    print_test_metrics,
)


def parse_img_size(val):
    if val is None:
        return None
    if isinstance(val, str) and "x" in val:
        h, w = val.lower().split("x")
        return (int(h), int(w))
    else:
        s = int(val)
        return (s, s)


def load_image_metadata(data_folder, test_df):
    """Load image metadata including original size and path"""
    image_info = {}

    for idx, row in test_df.iterrows():
        image_id = str(row["image_id"])
        image_path = os.path.join(data_folder, row["link"])

        # Get original size
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                original_size = (img.height, img.width)
        except:
            original_size = (448, 448)  # fallback

        # Get bboxes
        gt_bboxes = row.get("bbox_list", None)

        image_info[image_id] = {
            "original_size": original_size,
            "image_path": image_path,
            "bbx_list": gt_bboxes,
        }

    return image_info


def save_full_model(
    model,
    model_type,
    output,
    model_filename,
    actual_input_size,
    gradcam_layer,
    test_metrics,
    device,
):
    """Save full model with metadata"""
    if isinstance(actual_input_size, int):
        actual_input_size = (actual_input_size, actual_input_size)

    dummy_input = torch.randn(1, 3, actual_input_size[0], actual_input_size[1]).to(
        device
    )

    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    model_to_save.eval()
    with torch.no_grad():
        start = time.time()
        _ = model_to_save(dummy_input)
        inference_time = time.time() - start

    model_name = type(model_to_save).__name__
    num_params = sum(p.numel() for p in model_to_save.parameters())

    full_model_dir = os.path.join(output, "models")
    os.makedirs(full_model_dir, exist_ok=True)
    full_model_path = os.path.join(full_model_dir, f"{model_filename}_full.pth")

    model_metadata = {
        "model": model_to_save,
        "model_type": model_type,
        "model_name": model_name,
        "num_params": num_params,
        "input_size": actual_input_size,
        "gradcam_layer": gradcam_layer,
        "test_metrics": test_metrics,
        "inference_time": inference_time,
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "num_patches": 1,
        "arch_type": "based",
    }

    try:
        torch.save(model_metadata, full_model_path)
        print(f"\n✅ Saved full model to: {full_model_path}")
        print(f"   Model name: {model_metadata['model_name']}")
        print(f"   Model type: {model_metadata['model_type']}")
        print(f"   Num params: {model_metadata['num_params']}")
        print(f"   Input size: {model_metadata['input_size']}")
        print(f"   GradCAM layer: {model_metadata['gradcam_layer']}")
        print(f"   Inference time: {model_metadata['inference_time']:.4f}s")
        print(f"   Test Accuracy: {model_metadata['test_metrics']['accuracy']:.2f}%")
        if model_metadata["test_metrics"].get("auc"):
            print(f"   Test AUC: {model_metadata['test_metrics']['auc']:.4f}")
        print(
            f"   Test Precision: {model_metadata['test_metrics'].get('precision', 0.0):.2f}%"
        )
        print(
            f"   Test Recall: {model_metadata['test_metrics'].get('recall', 0.0):.2f}%"
        )
    except Exception as e:
        print(f"⚠️ Error saving full model: {e}")


def generate_visualizations(
    model,
    test_loader,
    test_df,
    image_info,
    class_names,
    output,
    model_filename,
    batch_size,
    actual_input_size,
    use_otsu,
    use_gradcam,
    gradcam_layer,
    device,
):
    """Generate visualizations for test set"""
    vis_dir = os.path.join(output, "test_based", model_filename)
    os.makedirs(vis_dir, exist_ok=True)

    test_image_ids = test_df["image_id"].tolist()
    model.eval()

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Visualizing")):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + len(images), len(test_image_ids))
        batch_image_ids = test_image_ids[start_idx:end_idx]

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        for i in range(len(images)):
            if i >= len(batch_image_ids):
                continue

            image_id = str(batch_image_ids[i])
            pred_class = predicted[i].item()
            gt_label = labels[i].item()
            pred_prob = probs[i, pred_class].item()

            if image_id not in image_info:
                continue

            info = image_info[image_id]
            original_size = info["original_size"]
            image_path = info["image_path"]
            gt_bboxes = info.get("bbx_list", None)

            # Generate GradCAM
            gradcam_map = None
            if use_gradcam and gradcam_layer:
                try:
                    input_tensor = images[i : i + 1].clone().requires_grad_(True)
                    test_model = (
                        model.module if isinstance(model, nn.DataParallel) else model
                    )
                    with torch.set_grad_enabled(True):
                        result = gradcam(
                            test_model,
                            input_tensor,
                            gradcam_layer,
                            class_idx=pred_class,
                        )
                        if (
                            isinstance(result, np.ndarray)
                            and result.ndim == 2
                            and result.dtype == np.uint8
                        ):
                            gradcam_map = result
                except Exception as e:
                    print(f"⚠️ GradCAM failed for {image_id}: {e}")

            save_path = os.path.join(vis_dir, f"{image_id}.png")
            try:
                visualize_classification_result(
                    images[i],
                    gt_bboxes,
                    pred_class,
                    gt_label,
                    pred_prob,
                    gradcam_map,
                    save_path,
                    class_names,
                    original_size,
                    image_path=image_path,
                    use_otsu=use_otsu,
                )
            except Exception as e:
                print(f"⚠️ Error visualizing {image_id}: {e}")

    print(f"✅ Saved visualizations to: {vis_dir}")


def run_based_test(
    data_folder,
    model_type,
    batch_size,
    output,
    config_path="config/config.yaml",
    img_size=None,
    pretrained_model_path=None,
    target_column=None,
    save_visualizations=True,
    only_viz=False,
    use_gradcam=True,
    use_otsu=True,  # Changed: default=True
):
    """Run classification model testing with GradCAM"""
    config = load_config(config_path)

    train_df, test_df, class_names = load_metadata(
        data_folder, config_path, target_column=target_column
    )

    if class_names and isinstance(class_names[0], int):
        class_names = [str(c) for c in class_names]

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Test samples: {len(test_df)}")

    # Load image metadata
    image_info = load_image_metadata(data_folder, test_df)

    _, test_loader = get_dataloaders(
        train_df,
        test_df,
        data_folder,
        batch_size=batch_size,
        config_path=config_path,
        img_size=img_size,
        mode="test",
    )

    model = get_based_model(model_type=model_type, num_classes=len(class_names))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not pretrained_model_path:
        print("⚠️ No pretrained model path provided!")
        return

    try:
        model.load_state_dict(
            torch.load(pretrained_model_path, map_location=device),  # trict=False
        )
        print(f"✅ Loaded pretrained model from {pretrained_model_path}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    try:
        sample_batch = next(iter(test_loader))
        images, _ = sample_batch
        actual_input_size = (images.shape[2], images.shape[3])
    except:
        actual_input_size = img_size if img_size else (448, 448)

    model_filename = os.path.basename(pretrained_model_path).replace(".pth", "")

    # GradCAM setup
    gradcam_layer = None
    if use_gradcam:
        test_model = model.module if isinstance(model, nn.DataParallel) else model
        gradcam_layer = get_gradcam_layer(test_model, model_type.lower())
        if gradcam_layer:
            print(f"✓ GradCAM enabled: {gradcam_layer}")
        else:
            use_gradcam = False
            print("⚠️ GradCAM layer not found")

    # Evaluation
    if not only_viz:
        print("\n" + "=" * 60)
        print("Evaluation on Test Set")
        print("=" * 60)

        results = evaluate_classification_model(model, test_loader, device)

        metrics = compute_classification_metrics(
            results["preds"], results["labels"], results["probs"], class_names
        )
        metrics["accuracy"] = results["accuracy"]

        print_test_metrics(metrics, class_names, results["labels"], results["preds"])

        save_full_model(
            model,
            model_type,
            output,
            model_filename,
            actual_input_size,
            gradcam_layer,
            metrics,
            device,
        )

    # Visualization
    if save_visualizations:
        print("\n" + "=" * 60)
        print("Generate Visualizations")
        print("=" * 60)

        generate_visualizations(
            model,
            test_loader,
            test_df,
            image_info,
            class_names,
            output,
            model_filename,
            batch_size,
            actual_input_size,
            use_otsu,
            use_gradcam,
            gradcam_layer,
            device,
        )

    if not only_viz:
        return metrics["accuracy"] / 100
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--img_size", type=str)
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--only_viz", action="store_true")
    parser.add_argument("--no_gradcam", action="store_true")
    parser.add_argument(
        "--no_otsu", action="store_true"
    )  # Changed: use --no_otsu to disable

    args = parser.parse_args()
    config = load_config(args.config)

    run_based_test(
        data_folder=get_arg_or_config(
            args.data_folder, config.get("data_folder"), None
        ),
        model_type=get_arg_or_config(args.model_type, config.get("model_type"), None),
        batch_size=get_arg_or_config(args.batch_size, config.get("batch_size"), 16),
        output=get_arg_or_config(args.output, config.get("output"), "output"),
        config_path=args.config,
        img_size=parse_img_size(
            get_arg_or_config(args.img_size, config.get("image_size"), None)
        ),
        pretrained_model_path=args.pretrained_model_path,
        target_column=get_arg_or_config(
            args.target_column, config.get("target_column"), None
        ),
        save_visualizations=not args.no_viz,
        only_viz=args.only_viz,
        use_gradcam=not args.no_gradcam,
        use_otsu=not args.no_otsu,  # Changed: invert logic
    )
