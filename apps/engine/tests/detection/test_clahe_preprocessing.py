from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from parkvision.detection.clahe_preprocessing import (
    clahe_grayscale,
    validate_clahe_parameters,
)


@pytest.mark.parametrize("grayscale", [False, True])
def test_clahe_matches_opencv_and_preserves_source_and_coordinates(
    grayscale: bool,
) -> None:
    # Non-square, non-tile-divisible image exercises OpenCV's border handling.
    bgr = np.random.default_rng(42).integers(0, 256, (61, 93, 3), dtype=np.uint8)
    gray = cast(NDArray[np.uint8], cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    image = gray if grayscale else bgr
    original = image.copy()
    image.flags.writeable = False

    processed = clahe_grayscale(image, clip_limit=3.0, tile_grid_size=(4, 7))

    assert processed.shape == image.shape[:2]
    assert processed.dtype == np.uint8
    assert not np.shares_memory(image, processed)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(
        processed, cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 7)).apply(gray)
    )
    # The unchanged BGR-only extractor will receive exactly these grayscale bytes.
    np.testing.assert_array_equal(
        cv2.cvtColor(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY),
        processed,
    )


@pytest.mark.parametrize("clip_limit", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_clip_limit(clip_limit: float) -> None:
    with pytest.raises(ValueError, match="clip_limit"):
        clahe_grayscale(np.zeros((10, 10), dtype=np.uint8), clip_limit=clip_limit)


@pytest.mark.parametrize(
    "grid",
    [(0, 8), (8, -1), (True, 8), (8.0, 8), (8,), (8, 8, 8), (2**31, 8)],
)
def test_rejects_invalid_tile_grid(grid: object) -> None:
    with pytest.raises(ValueError, match="tile_grid_size"):
        validate_clahe_parameters(2.0, cast(tuple[int, int], grid))


@pytest.mark.parametrize(
    "shape", [(0, 10), (10, 0), (0, 10, 3), (10,), (10, 10, 1), (10, 10, 4)]
)
def test_rejects_unsupported_or_empty_image(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        clahe_grayscale(np.zeros(shape, dtype=np.uint8))


def test_rejects_non_uint8() -> None:
    image = cast(NDArray[np.uint8], np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(TypeError, match="uint8"):
        clahe_grayscale(image)
