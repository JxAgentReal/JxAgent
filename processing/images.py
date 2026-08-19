"""Image processing: resize (never upscale), WebP encoding, small-target
preservation, and coordinate-consistent resizing (spec sections 17).

GUI screenshots are encoded losslessly; natural-photo replay data may remain
high-quality lossy WebP. Standard samples max long side 1600 px; difficult
grounding/action samples may keep up to 1920 px when a small target would
otherwise shrink below ~10-12 px.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image

from .coordinates import scale_point

WEBP_QUALITY = 80
DEFAULT_MAX_LONG = 1600
GROUNDING_MAX_LONG = 1920
MIN_TARGET_PX = 11  # keep target >= ~11 px after resize, else allow 1920


@dataclass
class ProcessedImage:
    data: bytes
    width: int
    height: int
    original_width: int
    original_height: int
    format: str = "WEBP"
    quality: int = WEBP_QUALITY
    lossless: bool = False

    @property
    def was_resized(self) -> bool:
        return (self.width, self.height) != (self.original_width, self.original_height)


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    # Training/evaluation parity: every decoded training image is RGB.
    # Keeping grayscale L images created a real distribution mismatch.
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def compute_target_long_side(original_size: Tuple[int, int],
                             target_width_px: Optional[float] = None,
                             target_height_px: Optional[float] = None,
                             min_target_px: int = MIN_TARGET_PX,
                             default_max_long: int = DEFAULT_MAX_LONG,
                             grounding_max_long: int = GROUNDING_MAX_LONG) -> int:
    """Choose a resize ceiling while preserving genuinely tiny targets.

    If a target's *smaller* dimension would shrink below ``min_target_px`` at
    the normal 1600px ceiling, solve the resize equation

        target_extent * new_long / source_long >= min_target_px

    for ``new_long``.  The previous implementation inverted this ratio and
    could retain too little resolution.  No path ever upscales the source.
    """
    import math
    w, h = int(original_size[0]), int(original_size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size: {original_size!r}")
    long_side = max(w, h)
    if long_side <= default_max_long:
        return default_max_long  # process_image still never upscales

    extents = [float(v) for v in (target_width_px, target_height_px)
               if v is not None and float(v) > 0]
    if not extents:
        return default_max_long
    target_extent = min(extents)
    projected = target_extent * default_max_long / long_side
    if projected >= min_target_px:
        return default_max_long

    needed_long = math.ceil(min_target_px * long_side / target_extent)
    return int(min(max(default_max_long, needed_long),
                   min(grounding_max_long, long_side)))


def resize_dimensions(width: int, height: int, max_long: int) -> Tuple[int, int]:
    long_side = max(width, height)
    if long_side <= max_long:
        return width, height
    scale = max_long / long_side
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def process_image(img: Image.Image,
                  max_long: int = DEFAULT_MAX_LONG,
                  quality: int = WEBP_QUALITY,
                  lossless: bool = False) -> ProcessedImage:
    """Resize without upscaling and encode WebP.

    GUI supervision should pass ``lossless=True`` so small text, icons and
    checkbox edges are not damaged by storage compression.
    """
    ow, oh = img.size
    nw, nh = resize_dimensions(ow, oh, max_long)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if (nw, nh) != (ow, oh):
        img = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    if lossless:
        img.save(buf, format="WEBP", lossless=True, method=6, exact=True)
    else:
        img.save(buf, format="WEBP", quality=quality, method=4)
    return ProcessedImage(data=buf.getvalue(), width=nw, height=nh,
                          original_width=ow, original_height=oh, quality=quality,
                          lossless=lossless)


def resize_point_with_image(x: float, y: float, original_size, new_size) -> Tuple[int, int]:
    """Scale a point consistent with image resizing; tests round-trip this."""
    return scale_point(x, y, original_size, new_size)


def save_image(data: bytes, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Read PNG IHDR dimensions without decoding the full image."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        import struct
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    return None
