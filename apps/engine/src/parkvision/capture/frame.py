from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Frame:
    camera_id: str
    sequence: int
    captured_at: datetime
    image: NDArray[np.uint8] = field(compare=False, hash=False, repr=False)
