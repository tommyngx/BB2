import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# Positive augmentations (bbox-safe, tăng cường đa dạng)
def get_positive_augmentations(height, width, enable_rotate90=True):
    crop_h = int(height * 0.80)  # Tăng crop để challenge hơn
    crop_w = int(width * 0.80)
    aug_list = [
        A.RandomSizedBBoxSafeCrop(
            height=crop_h, width=crop_w, erosion_rate=0.0, p=0.5
        ),  # Tăng p và erosion
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),  # Mở rộng scale
            translate_percent=(-0.15, 0.15),
            rotate=(-60, 60),  # Tăng rotate limit
            shear=(-20, 20),
            p=0.6,  # Tăng p
        ),
        A.ShiftScaleRotate(
            shift_limit=0.15,  # Tăng shift
            scale_limit=0.25,
            rotate_limit=45,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6
        ),  # Tăng limit và p
        A.RandomGamma(gamma_limit=(60, 140), p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.Equalize(p=0.3),
        A.OneOf(
            [
                A.Sharpen(alpha=(0.3, 0.6), lightness=(0.6, 1.6), p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ],
            p=0.4,  # Tăng p
        ),
        A.UnsharpMask(
            blur_limit=(3, 9), sigma_limit=(0.0, 4.0), alpha=(0.3, 0.6), p=0.3
        ),
        A.GaussNoise(var_limit=(20, 60), p=0.3),  # Tăng noise
        A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.2, 0.6), p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.OpticalDistortion(distort_limit=0.4, shift_limit=0.4, p=0.3),
        A.OneOf(
            [
                A.Downscale(
                    scale_min=0.7,
                    scale_max=0.9,  # Thay range để đa dạng
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=1.0,
                ),
            ],
            p=0.2,
        ),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4
        ),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
        A.ChannelShuffle(p=0.2),
        A.RandomFog(
            fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2
        ),  # Thêm weather effect, bbox-safe
        A.RandomRain(
            blur_value=3, brightness_coefficient=0.9, p=0.15
        ),  # Thêm rain, bbox-safe
        A.CoarseDropout(
            max_holes=8, max_height=height // 20, max_width=width // 20, p=0.3
        ),  # Thêm dropout bbox-safe
    ]
    if enable_rotate90:
        aug_list.insert(3, A.RandomRotate90(p=0.6))  # Tăng p
    return aug_list


# Negative augmentations (aggressive nhưng cân bằng hơn, thêm thực tế)
def get_negative_augmentations(height, width, enable_rotate90=True):
    aug_list = [
        A.CropAndPad(
            percent=(-0.25, 0.0), pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, p=0.5
        ),  # Align p with positive crop
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),  # Align with positive
            translate_percent=(-0.15, 0.15),
            rotate=(-60, 60),
            shear=(-20, 20),
            p=0.6,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.15, scale_limit=0.25, rotate_limit=45, p=0.5
        ),  # Align with positive
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6
        ),  # Align
        A.RandomGamma(gamma_limit=(60, 140), p=0.4),  # Align
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),  # Align
        A.Equalize(p=0.3),  # Align
        A.OneOf(
            [
                A.Sharpen(alpha=(0.3, 0.6), lightness=(0.6, 1.6), p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ],
            p=0.4,  # Align
        ),
        A.UnsharpMask(
            blur_limit=(3, 9), sigma_limit=(0.0, 4.0), alpha=(0.3, 0.6), p=0.3
        ),  # Align
        A.GaussNoise(var_limit=(20, 60), p=0.3),  # Align
        A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.2, 0.6), p=0.2),  # Align
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),  # Align p
        A.OpticalDistortion(distort_limit=0.4, shift_limit=0.4, p=0.3),  # Align p
        A.OneOf(
            [
                A.Downscale(
                    scale_min=0.7,
                    scale_max=0.9,
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=1.0,
                ),
            ],
            p=0.2,  # Align
        ),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4
        ),  # Align
        A.RGBShift(
            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3
        ),  # Align
        A.ChannelShuffle(p=0.2),  # Align
        A.RandomFog(
            fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2
        ),  # Add to match positive
        A.RandomRain(
            blur_value=3, brightness_coefficient=0.9, p=0.15
        ),  # Add to match positive
        A.CoarseDropout(
            max_holes=10, max_height=height // 15, max_width=width // 15, p=0.3
        ),  # Keep stronger dropout for negative
        A.ElasticTransform(alpha=1.5, sigma=40, p=0.2),  # Keep special for negative
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, p=0.2),  # Keep
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.15
        ),  # Keep
    ]
    if enable_rotate90:
        aug_list.insert(
            3, A.RandomRotate90(p=0.6)
        )  # Align insert position with positive
    return aug_list


# Compose for positive (with bbox)
def get_train_augmentation_positive(height, width, enable_rotate90=True):
    big_h = int(height * 1.3)  # Tăng resize để đa dạng hơn
    big_w = int(width * 1.3)
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
            min_area=100.0,  # Tăng để filter bbox nhỏ
            min_visibility=0.4,
        ),
    )


# Compose for negative (no bbox)
def get_train_augmentation_negative(height, width, enable_rotate90=True):
    big_h = int(height * 1.3)  # Align with positive
    big_w = int(width * 1.3)
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
