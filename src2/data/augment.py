import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# Positive augmentations (bbox-safe, align tương ứng với negative mới)
def get_positive_augmentations(height, width, enable_rotate90=True):
    crop_h = int(height * 0.80)
    crop_w = int(width * 0.80)
    aug_list = [
        A.RandomSizedBBoxSafeCrop(
            height=crop_h, width=crop_w, erosion_rate=0.0, p=0.3
        ),  # Bbox-safe crop, align p với negative CropAndPad
        A.OneOf(
            [
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LANCZOS4,
                    },
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.95, 0.95),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.8,
                ),
            ],
            p=0.125,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.01, 0.08),
            hole_width_range=(0.01, 0.15),
            fill=0,
            p=0.1,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0, 0.1),
            rotate=(-30, 30),
            shear=(-10, 10),
            p=0.3,
        ),
        # Bỏ ElasticTransform vì không bbox-safe, thay bằng GridDistortion nhẹ nếu cần, nhưng align bằng cách giữ nguyên các transform safe
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Equalize(p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
        A.UnsharpMask(
            blur_limit=(3, 5), sigma_limit=(1.0, 2.0), alpha=(0.1, 0.3), p=0.2
        ),
        A.GaussNoise(p=0.1),
    ]
    if enable_rotate90:
        aug_list.insert(5, A.RandomRotate90(p=0.5))  # Align insert position
    return aug_list


# Negative augmentations (sửa theo danh sách mới, giữ CropAndPad ở đầu)
def get_negative_augmentations(height, width, enable_rotate90=True):
    aug_list = [
        A.CropAndPad(
            percent=(-0.20, 0.0), pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, p=0.3
        ),
        A.OneOf(
            [
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LANCZOS4,
                    },
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.95, 0.95),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.8,
                ),
            ],
            p=0.125,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.01, 0.08),
            hole_width_range=(0.01, 0.15),
            fill=0,
            p=0.1,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0, 0.1),
            rotate=(-30, 30),
            shear=(-10, 10),
            p=0.3,
        ),
        A.ElasticTransform(alpha=1, sigma=20, p=0.1),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Equalize(p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
        A.UnsharpMask(
            blur_limit=(3, 5), sigma_limit=(1.0, 2.0), alpha=(0.1, 0.3), p=0.2
        ),
        A.GaussNoise(p=0.1),
    ]
    if enable_rotate90:
        aug_list.insert(5, A.RandomRotate90(p=0.5))  # Insert after VerticalFlip
    return aug_list


# Compose for positive (with bbox)
def get_train_augmentation_positive(height, width, enable_rotate90=True):
    big_h = int(height * 1.2)
    big_w = int(width * 1.2)
    aug_list = [A.Resize(big_h, big_w)]
    aug_list += get_positive_augmentations(
        height, width, enable_rotate90=enable_rotate90
    )
    aug_list += [
        A.Resize(height, width),
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ToTensorV2(),
    ]
    return A.Compose(
        aug_list,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_area=100.0,
            min_visibility=0.4,
        ),
    )


# Compose for negative (no bbox)
def get_train_augmentation_negative(height, width, enable_rotate90=True):
    big_h = int(height * 1.2)
    big_w = int(width * 1.2)
    aug_list = [A.Resize(big_h, big_w)]
    aug_list += get_negative_augmentations(
        height, width, enable_rotate90=enable_rotate90
    )
    aug_list += [
        A.Resize(height, width),
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ToTensorV2(),
    ]
    return A.Compose(aug_list)


# Test augmentations (giữ nguyên, nhưng normalize consistent)
def get_test_augmentation_positive(height, width):
    aug_list = [
        A.Resize(height, width),
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ToTensorV2(),
    ]
    return A.Compose(
        aug_list,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )


def get_test_augmentation_negative(height, width):
    aug_list = [
        A.Resize(height, width),
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ToTensorV2(),
    ]
    return A.Compose(aug_list)
