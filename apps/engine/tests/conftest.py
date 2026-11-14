import cv2
import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def geometry_row_image() -> NDArray[np.uint8]:
    image = np.full((260, 440, 3), 35, dtype=np.uint8)
    paint = (230, 230, 230)
    cv2.line(image, (42, 55), (390, 55), paint, 5)
    cv2.line(image, (33, 210), (405, 210), paint, 5)
    for near, far in zip((85, 160, 240, 321), (60, 151, 250, 355), strict=True):
        cv2.line(image, (near, 55), (far, 210), paint, 5)
    return image


@pytest.fixture
def geometry_uneven_row_image() -> NDArray[np.uint8]:
    image = np.full((300, 520, 3), 35, dtype=np.uint8)
    paint = (230, 230, 230)
    cv2.line(image, (30, 60), (470, 60), paint, 5)
    cv2.line(image, (25, 240), (482, 240), paint, 5)
    for front, rear in zip((80, 175, 285, 405), (48, 154, 300, 451), strict=True):
        cv2.line(image, (front, 60), (rear, 240), paint, 5)
    return image
