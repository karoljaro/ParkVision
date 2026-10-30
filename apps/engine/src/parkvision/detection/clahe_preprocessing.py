"""Opt-in image preprocessing for extraction experiments, independent of LSD."""

from math import isfinite
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def validate_clahe_parameters(
    clip_limit: float, tile_grid_size: tuple[int, int]
) -> None:
    """Require contrast limiting and a positive OpenCV-compatible tile count."""
    if not isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit must be finite and positive")
    if len(tile_grid_size) != 2 or any(
        type(size) is not int or not 0 < size <= 2**31 - 1 for size in tile_grid_size
    ):
        raise ValueError("tile_grid_size must contain two positive int32 tile counts")


def clahe_grayscale(
    image: NDArray[np.uint8],
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> NDArray[np.uint8]:
    """Return a new uint8 grayscale image, preserving size and pixel coordinates.

    Accept BGR or grayscale; BGR uses the same BGR2GRAY conversion as the current
    extractor. tile_grid_size counts tiles, not pixels. No blur or resize is used.
    """
    validate_clahe_parameters(clip_limit, tile_grid_size)
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError("Expected grayscale or BGR image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Expected a non-empty image")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cast(
        NDArray[np.uint8],
        cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(gray),
    )
