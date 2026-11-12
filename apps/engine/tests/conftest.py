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
