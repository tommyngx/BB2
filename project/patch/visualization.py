import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import random
import os


def plot_gradcam_plus(
    model,
    df,
    root_dir,
    num_images=4,
    random_state=None,
    input_size=448,
    save_dir=None,
    dataset_name=None,
    device=None,  # thêm device
):
    """
    Visualize Grad-CAM++ heatmaps for a batch of images using a given model.

    Args:
        model: Pretrained PyTorch model
        df: DataFrame containing image paths and labels
        root_dir: Root directory for image paths
        num_images: Number of images to visualize (default: 4)
        random_state: Random seed for reproducibility (default: None)
        input_size: Input image size (default: 448)
        save_dir: Directory to save Grad-CAM++ images (default: None)
    """
    # Set random seed
    if random_state is not None:
        random.seed(random_state)

    # Select random images
    indices = random.sample(list(df.index), num_images)
    img_paths = [os.path.join(root_dir, df.loc[idx, "link"]) for idx in indices]
    labels = [df.loc[idx, "cancer"] for idx in indices]

    # Set model to evaluation mode and select target layer
    model.eval()
    model_type = (
        model.__class__.__name__ if hasattr(model, "__class__") else str(type(model))
    )
    model_name = getattr(model, "model_name", str(model).lower())
    # print(model_type, "xxxx", model_name)

    if "DinoVisionTransformerClassifier" in model_type:
        target_layer = model.transformer.blocks[-1]
        default_input_size = 224
    elif "ResNet" in model_type:
        target_layer = model.layer4[-1]
        default_input_size = 224
    elif "FasterViT" in model_type:
        # target_layer = model.stages[-1][-1]
        target_layer = model.levels[1].blocks[-1].conv2
        default_input_size = 448
    # elif 'ConvNeXtV2' in model_type:
    #    target_layer = model.stages[-1][-1]
    #    default_input_size = 224
    elif "convnextv2" in model_name.lower() or "ConvNeXt" in model_type:
        # target_layer = model.stages[-1][-1]
        target_layer = model.stages[-1].downsample[0]
        default_input_size = 224
    # elif 'efficientnetv2' in model_type.lower():
    # elif 'efficientnetv2' in model_name or 'EfficientNet' in model_type:
    #    target_layer = model.features[-1]
    #    default_input_size = 448
    elif "efficientnetv2" in model_name or "EfficientNet" in model_type:
        # Verify model structure
        if hasattr(model, "blocks"):
            target_layer = model.blocks[-1]  # Last block for EfficientNetV2
        else:
            raise AttributeError(
                f"Model {model_type} does not have 'blocks' attribute. Check model architecture."
            )
        default_input_size = 448
    else:
        raise ValueError("Unsupported model type for Grad-CAM++")

    # Use provided input_size or default
    input_size = input_size if input_size is not None else default_input_size

    # Image preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    # Create figure
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 5 * num_images), dpi=100)
    if num_images == 1:
        axs = [axs]

    # Chuyển model về đúng device
    if device is None:
        device = (
            next(model.parameters()).device
            if next(model.parameters(), None) is not None
            else "cpu"
        )
    model = model.to(device)

    for i, (img_path, label) in enumerate(zip(img_paths, labels)):
        # Load and preprocess image
        img_pil = Image.open(img_path).convert("RGB")
        orig_size = img_pil.size
        input_tensor = preprocess(img_pil).unsqueeze(0).requires_grad_(True)
        input_tensor = input_tensor.to(device)  # chuyển input về đúng device

        # Grad-CAM++ implementation
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hooks
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass and compute gradients
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        score = output[0, pred_class]
        score.backward()

        # Compute Grad-CAM++ heatmap
        fmap = activations[0][0]
        grad = gradients[0][0]
        alpha_num = grad**2
        alpha_denom = 2 * grad**2 + torch.sum(fmap * grad**3, dim=(1, 2), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-8)
        weights = torch.sum(alpha * torch.relu(grad), dim=(1, 2))

        cam = torch.zeros(fmap.shape[1:], dtype=torch.float32, device=fmap.device)
        for j, w in enumerate(weights):
            cam += w * fmap[j]

        # Process heatmap
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() != 0 else cam
        cam = np.uint8(cam * 255)
        cam_img = Image.fromarray(cam).resize(orig_size, Image.Resampling.LANCZOS)
        cam_img_np = np.array(cam_img)

        # Prepare original image
        img_np = np.array(img_pil)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        # Plot original image
        axs[i][0].imshow(img_np)
        axs[i][0].set_title(f"Original Image\nLabel: {label}", fontsize=12, pad=10)
        axs[i][0].axis("off")

        # Plot image with Grad-CAM++ overlay
        axs[i][1].imshow(img_np)
        axs[i][1].imshow(cam_img_np, cmap="jet", alpha=0.5)
        axs[i][1].set_title(
            f"Grad-CAM++\nPrediction: {pred_class}", fontsize=12, pad=10
        )
        axs[i][1].axis("off")

        # Remove hooks
        handle_fwd.remove()
        handle_bwd.remove()

    plt.tight_layout(pad=2.0)
    # Save the whole figure if save_dir is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if dataset_name is not None:
            dataset_key = str(dataset_name)
        else:
            dataset_key = os.path.basename(os.path.normpath(root_dir))
        model_type_name = (
            model.__class__.__name__
            if hasattr(model, "__class__")
            else str(type(model))
        )
        model_key = f"{dataset_key}_{model_type_name}".replace(" ", "")
        save_path = os.path.join(save_dir, f"{model_key}_gradcam.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved GradCAM visualization to {save_path}")
    # plt.show()
