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
    """
    Split an image into vertical patches with optional overlap.
    Output same type and shape as input (PIL.Image, np.ndarray, torch.Tensor).
    Last patch always taken from bottom up, no padding added.

    Parameters
    ----------
    img : PIL.Image.Image or np.ndarray or torch.Tensor
        Input image to split.
    num_patches : int
        Number of vertical patches to create (e.g., 2, 4).
    patch_size : tuple of int, optional
        Not used in this implementation. Kept for compatibility.

    Returns
    -------
    patches : list of PIL.Image.Image or list of np.ndarray or list of torch.Tensor
        List of patch images with same type as input.

    Notes
    -----
    - Image is divided into vertical patches with 20% overlap
    - Last patch is always taken from bottom to top
    - All patches have the same height
    - Does NOT include global image (that's handled in pre_mil_gradcam)
    """
    overlap_ratio = 0.2

    # Remember input type and original shape
    input_type = type(img)
    orig_shape = None

    # Convert to numpy for processing
    if isinstance(img, torch.Tensor):
        orig_shape = img.shape  # (C, H, W)
        image = img.permute(1, 2, 0).cpu().numpy()
    elif isinstance(img, Image.Image):
        image = np.array(img)
    elif isinstance(img, np.ndarray):
        orig_shape = img.shape
        image = img
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")

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
        patches.append(patch)

    # Convert back to original input type
    if input_type is torch.Tensor:
        # (H, W, C) -> (C, H, W)
        patches = [torch.from_numpy(p).permute(2, 0, 1).float() for p in patches]
    elif input_type is Image.Image:
        patches = [Image.fromarray(p) for p in patches]
    # If np.ndarray, keep as is

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
    """
    Prepare data and perform inference before GradCAM visualization.

    This function loads and preprocesses an image, runs inference to get
    predictions, and prepares all necessary components for GradCAM generation.
    Supports both standard models and MIL/patch-based models.

    Parameters
    ----------
    model_tuple : tuple
        A tuple containing model and metadata in the following order:
        - model : torch.nn.Module
            The neural network model.
        - input_size : tuple of int
            Expected input size as (height, width).
        - model_name : str or None
            Name of the model architecture.
        - gradcam_layer : str or None
            Default target layer for GradCAM.
        - normalize : dict or None
            Normalization parameters with 'mean' and 'std' keys.
        - num_patches : int or None
            Number of patches for MIL models. None for standard models.
        - arch_type : str or None
            Architecture type (e.g., 'mil', 'mil_v4'). None for standard models.
    image_path : str
        Path to the input image file.
    target_layer : str, optional
        Name of the layer to generate GradCAM from. If None, uses the
        gradcam_layer from model_tuple or defaults to 'layer4'.
        Default is None.
    class_idx : int, optional
        Index of the class to visualize. If None, uses the predicted
        class with highest probability. Default is None.

    Returns
    -------
    model : torch.nn.Module
        The neural network model in evaluation mode.
    input_tensor : torch.Tensor
        Preprocessed input tensor. Shape [1, 3, H, W] for standard models,
        or [1, N, 3, H, W] or [1, N+1, 3, H, W] for MIL models.
    img : PIL.Image.Image
        Original input image in RGB format.
    target_layer : str
        Name of the target layer to be used for GradCAM.
    class_idx : int or None
        Input class index (passes through the input parameter).
    pred_class : int
        Predicted class index (0-based integer).
    pred_prob : float
        Prediction probability for the target class, ranging from 0.0 to 1.0.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist at the specified path.

    Notes
    -----
    - The image is automatically converted to RGB format.
    - For standard models: preprocessing includes resizing, converting to tensor,
      and normalization.
    - For MIL models: the image is split into patches, and an optional global
      image is added (for v4 architecture).
    - The model is set to evaluation mode and inference is done without
      gradient computation.
    - If class_idx is None, the function automatically selects the class
      with the highest prediction probability.

    Examples
    --------
    >>> from model_loader import load_full_model
    >>> # Standard model
    >>> model_tuple = load_full_model('checkpoint.pth')
    >>> results = pre_gradcam(model_tuple, 'image.jpg')
    >>> model, input_tensor, img, layer, _, pred_class, prob = results
    >>> print(f"Predicted class: {pred_class}, Probability: {prob:.2%}")
    Predicted class: 2, Probability: 87.35%

    >>> # MIL model
    >>> mil_tuple = load_full_model('mil_model.pth')
    >>> results = pre_gradcam(mil_tuple, 'image.jpg')
    >>> model, input_tensor, img, layer, _, pred_class, prob = results
    >>> print(f"Input shape for MIL: {input_tensor.shape}")
    Input shape for MIL: torch.Size([1, 4, 3, 448, 448])
    """
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

        print(
            f"DEBUG: Split into {len(patches)} vertical patches (num_patches={num_patches})"
        )

        # Preprocess each patch - ensure each patch is resized to input_size
        preprocess = transforms.Compose(
            [
                transforms.Resize(resize_size),  # Resize each patch to input_size
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        patch_tensors = [preprocess(patch) for patch in patches]

        # Stack patches: shape [N, 3, H, W]
        patches_tensor = torch.stack(patch_tensors, dim=0)
        print(f"DEBUG: Stacked patches tensor shape: {patches_tensor.shape}")

        # Check if arch_type requires global image (like mil_v4)
        if "v4" in arch_type.lower() or "global" in arch_type.lower():
            # Add global image as the last patch
            # resize_size is (H, W), PIL.resize expects (W, H)
            resize_size_pil = (resize_size[1], resize_size[0])
            global_img = img.resize(resize_size_pil, Image.Resampling.BILINEAR)
            global_tensor = preprocess(global_img)
            # Concatenate: [N+1, 3, H, W]
            patches_tensor = torch.cat(
                [patches_tensor, global_tensor.unsqueeze(0)], dim=0
            )
            print(
                f"DEBUG: Added global image, final patches tensor shape: {patches_tensor.shape}"
            )

        # Add batch dimension: [1, N, 3, H, W] or [1, N+1, 3, H, W]
        input_tensor = patches_tensor.unsqueeze(0)
        print(f"DEBUG: Final input_tensor shape: {input_tensor.shape}")
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
        - For MIL models: shape [num_input_patches, H, W] - one heatmap per INPUT patch
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

    # Get number of input patches from input_tensor
    num_input_patches = input_tensor.shape[1] if input_tensor.dim() == 5 else 1
    print(f"DEBUG: num_input_patches from input = {num_input_patches}")
    print(f"DEBUG: grads.shape = {grads.shape}, acts.shape = {acts.shape}")

    # Process based on activation dimensions
    if grads.dim() == 5:  # [B, N_out, C, H, W]
        B, N_out, C, H, W = grads.shape
        print(f"DEBUG: Model has {N_out} feature patches in activations")

        # Calculate heatmap for each activation patch
        cams = []
        for i in range(N_out):
            g = grads[0, i]  # [C, H, W]
            a = acts[0, i]  # [C, H, W]
            weights = g.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
            cam = (weights * a).sum(dim=0)  # [H, W]
            cam = torch.relu(cam)
            cams.append(cam)

        # If model aggregated patches (N_out < num_input_patches), distribute heatmaps
        if N_out < num_input_patches:
            print(
                f"⚠️ Model aggregated {num_input_patches} patches into {N_out} features"
            )
            print(
                f"   Distributing {N_out} heatmaps to {num_input_patches} input patches"
            )

            # Strategy: Assign first N_out-1 heatmaps to first N_out-1 patches,
            # then duplicate last heatmap for remaining patches
            final_cams = cams[:N_out]

            # Add duplicates of the last heatmap for remaining patches
            while len(final_cams) < num_input_patches:
                final_cams.append(cams[-1].clone())

            cam = torch.stack(final_cams, dim=0)  # [num_input_patches, H, W]
        else:
            cam = torch.stack(cams[:num_input_patches], dim=0)

    elif grads.dim() == 4:  # [B, C, H, W] - fully aggregated
        print(f"DEBUG: Model fully aggregated all patches")
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam_single = (weights * acts).sum(dim=1, keepdim=True)
        cam_single = torch.relu(cam_single).squeeze()  # [H, W]

        # Replicate for all input patches
        cam = torch.stack([cam_single] * num_input_patches, dim=0)
    else:
        print(f"⚠️ Unexpected grads.dim() = {grads.dim()}")
        # Fallback
        while grads.dim() > 4:
            grads = grads.mean(dim=1)
            acts = acts.mean(dim=1)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze()
        if cam.ndim == 2:
            cam = torch.stack([cam] * num_input_patches, dim=0)

    # Convert to numpy
    cam = cam.cpu().numpy()
    print(f"DEBUG: cam.shape after processing: {cam.shape}")

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
    else:
        # Should not reach here if logic above is correct
        if cam.ndim == 0:
            cam = np.array([[cam]])
        elif cam.ndim == 1:
            size = int(np.sqrt(cam.size))
            if size * size == cam.size:
                cam = cam.reshape(size, size)
            else:
                cam = cam.reshape(1, -1)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        cam = np.uint8(cam * 255)

    print(f"DEBUG: Final cam.shape: {cam.shape}, dtype: {cam.dtype}")

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
    """
    Visualize GradCAM heatmap with multiple display options.

    This function provides flexible visualization of GradCAM heatmaps overlaid
    on original images, with support for bounding boxes, predictions, and
    various display layouts.

    Parameters
    ----------
    cam : numpy.ndarray
        GradCAM heatmap array with shape [H, W] and dtype uint8.
        Values should be in range [0, 255].
    img : PIL.Image.Image
        Original input image in RGB format.
    option : int, optional
        Visualization mode (1-5). Default is 1.

        - 1 : Original image with heatmap overlay
        - 2 : Side-by-side display (original | heatmap)
        - 3 : Three-panel display (original | heatmap | blended)
        - 4 : Three-panel with Otsu filtering (original | heatmap | blended with mask)
        - 5 : Four-panel display (original | heatmap | blended | blended with Otsu mask)
    blend_alpha : float, optional
        Blending factor for heatmap overlay, ranging from 0.0 to 1.0.
        Higher values make the heatmap more opaque. Used in options 3, 4, and 5.
        Default is 0.5.
    bbx_list : list of list of int, optional
        List of bounding boxes to draw on the original image.
        Each bounding box is formatted as [x, y, width, height] where:

        - x, y : top-left corner coordinates
        - width, height : box dimensions

        Boxes are drawn with lime green borders. Default is None.
    pred : str, optional
        Predicted class name to display in the title. Default is None.
    prob : float, optional
        Prediction probability (0.0 to 1.0) to display in the title as percentage.
        Default is None.
    gt_label : str, optional
        Ground truth label to display in the title. Default is None.

    Returns
    -------
    None
        The function displays a matplotlib figure and does not return a value.

    Raises
    ------
    ValueError
        If option is not in the range [1, 5].

    Notes
    -----
    - The heatmap is resized to match the original image dimensions using
      bilinear interpolation.
    - The 'jet' colormap is used for heatmap visualization (blue=low, red=high).
    - Otsu thresholding (options 4 and 5) automatically determines an optimal
      threshold to highlight only the most important regions.
    - Title format: "Original Image,|GT: {gt_label}|Pred: {pred}|,Prob: {prob}%"
    - All axes are hidden for cleaner visualization.

    Examples
    --------
    >>> from model_loader import load_full_model
    >>> from gradcam_utils import pre_gradcam, gradcam_plus_plus, post_gradcam
    >>>
    >>> # Basic usage with option 1
    >>> model_tuple = load_full_model('checkpoint.pth')
    >>> model, input_tensor, img, layer, _, pred_class, prob = pre_gradcam(model_tuple, 'cat.jpg')
    >>> cam = gradcam_plus_plus(model, input_tensor, layer)
    >>> post_gradcam(cam, img, option=1)

    >>> # With prediction and probability
    >>> post_gradcam(cam, img, option=2, pred="Cat", prob=0.873)

    >>> # With bounding boxes
    >>> bboxes = [[50, 50, 200, 150], [300, 100, 150, 200]]
    >>> post_gradcam(cam, img, option=3, bbx_list=bboxes, pred="Dog", prob=0.921)

    >>> # Full annotation with ground truth
    >>> post_gradcam(cam, img, option=5, blend_alpha=0.6,
    ...              bbx_list=bboxes, pred="Cat", prob=0.85, gt_label="Cat")

    >>> # Otsu filtered visualization for cleaner focus
    >>> post_gradcam(cam, img, option=4, blend_alpha=0.7, pred="Bird", prob=0.78)
    """
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
    """
    Generate GradCAM++ (improved GradCAM) heatmap.

    GradCAM++ is an improved version of GradCAM that provides better
    visual explanations, especially for images with multiple occurrences
    of the same class. It uses a more sophisticated weighting scheme
    based on higher-order gradients.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to visualize.
    input_tensor : torch.Tensor
        Input tensor with shape [1, 3, H, W] where H and W are height
        and width of the input image.
    target_layer : str
        Name of the convolutional layer to generate the CAM from
        (e.g., 'layer4', 'features.7').
    class_idx : int, optional
        Index of the target class to visualize. If None, uses the
        predicted class with highest score. Default is None.

    Returns
    -------
    cam : numpy.ndarray
        GradCAM++ heatmap with shape [H', W'] where H' and W' are the
        spatial dimensions of the target layer's feature map.
        Values are normalized to range [0, 255] as uint8.

    Notes
    -----
    - GradCAM++ computes pixel-wise weights using the formula:
      α^kc_ij = (∂²Y^c/∂A^k_ij²) / (2(∂²Y^c/∂A^k_ij²) + Σ_ab(A^k_ab)(∂³Y^c/∂A^k_ij³))
    - These weights better capture the importance of different spatial
      locations compared to standard GradCAM.
    - More accurate for localizing multiple instances of the same object.
    - Numerical stability is ensured by adding epsilon (1e-8) to denominators.
    - Hooks are automatically removed after computation.

    References
    ----------
    .. [1] Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based Visual
       Explanations for Deep Convolutional Networks." WACV 2018.

    Examples
    --------
    >>> from model_loader import load_full_model
    >>> model_tuple = load_full_model('checkpoint.pth')
    >>> model, input_tensor, img, layer, _, _, _ = pre_gradcam(model_tuple, 'dogs.jpg')
    >>> heatmap = gradcam_plus_plus(model, input_tensor, layer)
    >>> print(f"Heatmap shape: {heatmap.shape}, max value: {heatmap.max()}")
    Heatmap shape: (14, 14), max value: 255

    >>> # Compare with standard GradCAM
    >>> heatmap_std = gradcam(model, input_tensor, layer)
    >>> heatmap_pp = gradcam_plus_plus(model, input_tensor, layer)
    >>> # GradCAM++ typically provides sharper, more accurate localization
    """
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
