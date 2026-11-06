from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropEmptyConfig:
    percentage_to_remove: float = 0.9
    remove_page_number: bool = False
    color_threshold: int = 240
    min_white_fraction: float = 0.99
    content_density_sides: float = 0.001
    content_density_main_text: float = 0.05
    content_density_any: float = 1e-6
    preserve_border_px: int = 1
    uniform_rowcol_std_threshold: float = 0.0


def crop_empty(
    image: Image.Image, *, config: CropEmptyConfig
) -> Tuple[Image.Image, Dict[str, Any]]:
    img = image.convert("RGB")
    arr = np.array(img)
    intensity = arr.mean(axis=2)

    def _find_border_start(axis: int, *, min_content_density_threshold: float) -> int:
        size = intensity.shape[axis]
        for i in range(size):
            pixels = intensity[i, :] if axis == 0 else intensity[:, i]
            white = float(np.mean(pixels > config.color_threshold))
            non_white = 1.0 - white
            if float(config.uniform_rowcol_std_threshold) > 0.0 and float(np.std(pixels)) <= float(
                config.uniform_rowcol_std_threshold
            ):
                continue
            if (white < config.min_white_fraction) and (non_white > min_content_density_threshold):
                return int(i)
        return int(size)

    def _find_border_end(axis: int, *, min_content_density_threshold: float) -> int:
        size = intensity.shape[axis]
        for i in range(size - 1, -1, -1):
            pixels = intensity[i, :] if axis == 0 else intensity[:, i]
            white = float(np.mean(pixels > config.color_threshold))
            non_white = 1.0 - white
            if float(config.uniform_rowcol_std_threshold) > 0.0 and float(np.std(pixels)) <= float(
                config.uniform_rowcol_std_threshold
            ):
                continue
            if (white < config.min_white_fraction) and (non_white > min_content_density_threshold):
                return int(i + 1)
        return 0

    top = _find_border_start(0, min_content_density_threshold=float(config.content_density_sides))
    left = _find_border_start(1, min_content_density_threshold=float(config.content_density_sides))
    right = _find_border_end(1, min_content_density_threshold=float(config.content_density_sides))

    main_text_end = _find_border_end(
        0, min_content_density_threshold=float(config.content_density_main_text)
    )
    last_content_end = _find_border_end(
        0, min_content_density_threshold=float(config.content_density_any)
    )
    bottom = main_text_end if config.remove_page_number else last_content_end

    width, height = img.size
    pad = max(int(getattr(config, "preserve_border_px", 0) or 0), 0)
    if pad > 0:
        left = max(int(left) - pad, 0)
        top = max(int(top) - pad, 0)
        right = min(int(right) + pad, int(width))
        bottom = min(int(bottom) + pad, int(height))
    crop_box = (int(left), int(top), int(right), int(bottom))
    valid = 0 <= crop_box[0] < crop_box[2] <= width and 0 <= crop_box[1] < crop_box[3] <= height

    if not valid:
        return image, {
            "applied": False,
            "crop_box": None,
            "original_width": int(width),
            "original_height": int(height),
            "cropped_width": int(width),
            "cropped_height": int(height),
            "config": {
                "percentage_to_remove": float(config.percentage_to_remove),
                "remove_page_number": bool(config.remove_page_number),
                "color_threshold": int(config.color_threshold),
                "min_white_fraction": float(config.min_white_fraction),
                "content_density_sides": float(config.content_density_sides),
                "content_density_main_text": float(config.content_density_main_text),
                "content_density_any": float(config.content_density_any),
                "preserve_border_px": int(config.preserve_border_px),
                "uniform_rowcol_std_threshold": float(config.uniform_rowcol_std_threshold),
            },
        }

    cropped = img.crop(crop_box)
    return cropped, {
        "applied": True,
        "crop_box": [int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3])],
        "original_width": int(width),
        "original_height": int(height),
        "cropped_width": int(cropped.width),
        "cropped_height": int(cropped.height),
        "config": {
            "percentage_to_remove": float(config.percentage_to_remove),
            "remove_page_number": bool(config.remove_page_number),
            "color_threshold": int(config.color_threshold),
            "min_white_fraction": float(config.min_white_fraction),
            "content_density_sides": float(config.content_density_sides),
            "content_density_main_text": float(config.content_density_main_text),
            "content_density_any": float(config.content_density_any),
            "preserve_border_px": int(config.preserve_border_px),
            "uniform_rowcol_std_threshold": float(config.uniform_rowcol_std_threshold),
        },
    }
