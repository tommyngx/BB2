"""GradCAM and GradCAM++ implementation utilities."""

import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import math
# from src.data.patch_dataset import split_image_into_patches


def split_image_into_patches(
    img: Image.Image, num_patches: int, patch_size: tuple[int, int] = None
) -> list[Image.Image]:
    overlap_ratio = 0.2

    # Convert PIL to numpy for processing
    image = np.array(img)

    # Ensure (H, W, C) format
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 3 and image.shape[-1] != 3:
        if image.shape[1] == 3:
            image = np.transpose(image, (0, 2, 1))

    height, width = image.shape[:2]
    patch_height = height // num_patches
    step = int(patch_height * (1 - overlap_ratio))

    if num_patches == 1 or step <= 0:
        starts = [0]
    else:
        starts = [i * step for i in range(num_patches - 1)]
        starts.append(height - patch_height)

    patches = []
    for i, start_h in enumerate(starts):
        end_h = start_h + patch_height
        # Last patch always taken from bottom up
        if i == num_patches - 1:
            start_h = height - patch_height
            end_h = height
        patch = image[start_h:end_h, :, :]
        patches.append(Image.fromarray(patch))

    return patches


def pre_mil_gradcam(
    model_tuple: tuple[
        nn.Module,
        tuple[int, int],
        str | None,
        str | None,
        dict[str, list[float]] | None,
    ],
    image_path: str,
    target_layer: str | None = None,
    class_idx: int | None = None,
) -> tuple[nn.Module, torch.Tensor, Image.Image, str, int | None, int, float]:
    # Unpack model and info
    model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = (
        model_tuple
    )
    model.eval()

    # Use gradcam_layer if available, else default to 'layer4'
    if target_layer is None:
        target_layer = gradcam_layer if gradcam_layer is not None else "layer4"

    # Use input_size directly as resize_size (expected to be a tuple like (448, 448))
    resize_size = input_size

    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path).convert("RGB")

    # Check if this is a MIL/patch-based model
    is_mil_model = num_patches is not None and arch_type is not None

    if is_mil_model:
        # For MIL models: split image into patches
        patches = split_image_into_patches(img, num_patches, resize_size)

        # Preprocess each patch
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        patch_tensors = [preprocess(patch) for patch in patches]

        # Stack patches: shape [N, 3, H, W]
        patches_tensor = torch.stack(patch_tensors, dim=0)

        # Check if arch_type requires global image (like mil_v4)
        if "v4" in arch_type.lower() or "global" in arch_type.lower():
            # Add global image as the last patch
            global_img = img.resize(resize_size, Image.Resampling.BILINEAR)
            global_tensor = preprocess(global_img)
            # Concatenate: [N+1, 3, H, W]
            patches_tensor = torch.cat(
                [patches_tensor, global_tensor.unsqueeze(0)], dim=0
            )

        # Add batch dimension: [1, N, 3, H, W] or [1, N+1, 3, H, W]
        input_tensor = patches_tensor.unsqueeze(0)
    else:
        # For standard models: single image preprocessing
        preprocess = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )
        input_tensor = preprocess(img).unsqueeze(0)

    # Predict class probabilities
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        # If class_idx is not specified, use the highest predicted class
        pred_class = (
            class_idx if class_idx is not None else int(output.argmax(dim=1).item())
        )
        pred_prob = probs[pred_class]

    return model, input_tensor, img, target_layer, class_idx, pred_class, pred_prob


def mil_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str,
    class_idx: int | None = None,
) -> ndarray:
    """
    Generate GradCAM for MIL/patch-based models.

    Returns
    -------
    cam : ndarray
        - For MIL models: shape [num_patches, H, W] - one heatmap per patch
        - For standard models: shape [H, W] - single heatmap
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    layer = dict([*model.named_modules()])[target_layer]

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]

    # For MIL models: keep instance dimension [B, N, C, H, W]
    # Process each instance separately to get per-patch heatmaps
    if grads.dim() == 5:  # [B, N, C, H, W]
        B, N, C, H, W = grads.shape
        cams = []
        for i in range(N):
            g = grads[0, i]  # [C, H, W]
            a = acts[0, i]  # [C, H, W]
            weights = g.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
            cam = (weights * a).sum(dim=0)  # [H, W]
            cam = torch.relu(cam)
            cams.append(cam)
        cam = torch.stack(cams, dim=0)  # [N, H, W]
    else:
        # Standard model: [B, C, H, W]
        while grads.dim() > 4:
            grads = grads.mean(dim=1)
            acts = acts.mean(dim=1)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze()  # Remove batch and channel dims

    # Convert to numpy
    cam = cam.cpu().numpy()

    # Normalize each patch heatmap independently
    if cam.ndim == 3:  # [N, H, W]
        normalized_cams = []
        for i in range(cam.shape[0]):
            c = cam[i]
            c_min, c_max = c.min(), c.max()
            if c_max > c_min:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = np.zeros_like(c)
            normalized_cams.append(np.uint8(c * 255))
        cam = np.stack(normalized_cams, axis=0)
    else:  # [H, W]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        cam = np.uint8(cam * 255)

    handle_f.remove()
    handle_b.remove()

    return cam


def post_mil_gradcam(
    cam: ndarray,
    img: Image.Image,
    option: int = 1,
    blend_alpha: float = 0.5,
    bbx_list: list[list[int]] | None = None,
    pred: str | None = None,
    prob: float | None = None,
    gt_label: str | None = None,
) -> None:
    print("inside post_mil_gradcam function")
    print("cam shape:", cam.shape)
    print("img size:", img.size)

    cam_img = Image.fromarray(cam).resize(img.size, resample=Image.Resampling.BILINEAR)
    cam_img_np = np.array(cam_img)

    def draw_bbx(ax, bbx_list):
        for bbx in bbx_list:
            x, y, w, h = map(int, bbx)
            rect = plt.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

    main_title = f"Original Image"
    if gt_label is not None:
        main_title += f",|GT: {gt_label}|"
    if pred is not None:
        main_title += f"Pred: {pred}|"
    if prob is not None:
        main_title += f",Prob: {prob * 100:.1f}%"

    if option == 1:
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(cam_img_np, cmap="jet", alpha=0.5)
        if bbx_list is not None:
            draw_bbx(ax, bbx_list)
        ax.set_title(main_title)
        ax.axis("off")
        plt.show()
    elif option == 2:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        if bbx_list is not None:
            draw_bbx(axs[0], bbx_list)
        axs[0].set_title(main_title)
        axs[0].axis("off")
        axs[1].imshow(cam_img_np, cmap="jet")
        axs[1].set_title("GradCAM Heatmap")
        axs[1].axis("off")
        plt.show()
    elif option == 3:
        blend = np.array(img).astype(np.float32) / 255.0
        cam_color = plt.cm.jet(cam_img_np / 255.0)[..., :3]
        blend_img = (1 - blend_alpha) * blend + blend_alpha * cam_color
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img)
        if bbx_list is not None:
            draw_bbx(axs[0], bbx_list)
        axs[0].set_title(main_title)
        axs[0].axis("off")
        axs[1].imshow(cam_img_np, cmap="jet")
        axs[1].set_title("GradCAM Heatmap")
        axs[1].axis("off")
        axs[2].imshow(blend_img)
        axs[2].set_title("Blended Image")
        axs[2].axis("off")
        plt.show()
    elif option == 4:
        otsu_thresh = threshold_otsu(cam_img_np)
        mask = cam_img_np > otsu_thresh
        blend = np.array(img).astype(np.float32) / 255.0
        cam_color = plt.cm.jet(cam_img_np / 255.0)[..., :3]
        blend_img = blend.copy()
        blend_img[mask] = (1 - blend_alpha) * blend_img[mask] + blend_alpha * cam_color[
            mask
        ]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img)
        if bbx_list is not None:
            draw_bbx(axs[0], bbx_list)
        axs[0].set_title(main_title)
        axs[0].axis("off")
        axs[1].imshow(cam_img_np, cmap="jet")
        axs[1].set_title("GradCAM Heatmap")
        axs[1].axis("off")
        axs[2].imshow(blend_img)
        axs[2].set_title("Blended Image (Otsu filtered)")
        axs[2].axis("off")
        plt.show()
    elif option == 5:
        blend = np.array(img).astype(np.float32) / 255.0
        cam_color = plt.cm.jet(cam_img_np / 255.0)[..., :3]
        blend_img = (1 - blend_alpha) * blend + blend_alpha * cam_color
        otsu_thresh = threshold_otsu(cam_img_np)
        mask = cam_img_np > otsu_thresh
        blend_img_otsu = blend.copy()
        blend_img_otsu[mask] = (1 - blend_alpha) * blend_img_otsu[
            mask
        ] + blend_alpha * cam_color[mask]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(img)
        if bbx_list is not None:
            draw_bbx(axs[0], bbx_list)
        axs[0].set_title(main_title)
        axs[0].axis("off")
        axs[1].imshow(cam_img_np, cmap="jet")
        axs[1].set_title("GradCAM Heatmap")
        axs[1].axis("off")
        axs[2].imshow(blend_img)
        axs[2].set_title("Blended Image")
        axs[2].axis("off")
        axs[3].imshow(blend_img_otsu)
        axs[3].set_title("Blended Image (Otsu filtered)")
        axs[3].axis("off")
        plt.show()
    else:
        raise ValueError("option must be 1, 2, 3, 4, or 5")


def mil_gradcam_plus_plus(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str,
    class_idx: int | None = None,
) -> ndarray:
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    layer = dict([*model.named_modules()])[target_layer]

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward(retain_graph=True)

    grads = gradients[0]  # [N, C, H, W]
    acts = activations[0]  # [N, C, H, W]

    # GradCAM++ weights calculation
    grads = grads[0]  # [C, H, W]
    acts = acts[0]  # [C, H, W]
    grads_power_2 = grads**2
    grads_power_3 = grads**3

    sum_acts = torch.sum(acts, dim=(1, 2), keepdim=True)  # [C, 1, 1]
    eps = 1e-8

    alpha_num = grads_power_2
    alpha_denom = grads_power_2 * 2.0 + sum_acts * grads_power_3
    alpha_denom = torch.where(
        alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps
    )
    alphas = alpha_num / alpha_denom  # [C, H, W]

    relu_grads = torch.relu(grads)
    weights = torch.sum(alphas * relu_grads, dim=(1, 2))  # [C]

    cam = torch.sum(weights.view(-1, 1, 1) * acts, dim=0)  # [H, W]
    cam = torch.relu(cam)
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.uint8(cam * 255)

    handle_f.remove()
    handle_b.remove()

    return cam
