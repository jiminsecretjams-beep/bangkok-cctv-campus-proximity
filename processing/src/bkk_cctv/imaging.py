"""A single orientation/color convention for quality, inference and labels."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
