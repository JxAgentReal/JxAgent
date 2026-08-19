#!/usr/bin/env python3
"""Shared observation preprocessing for base and adapter arms.

ONE function used by every arm so the model-visible image can never differ
between base and adapter. Coordinates emitted by the model live in the
PROCESSED (observed) image space; executed coordinates are mapped back into
environment space with the recorded inverse transform.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402


@dataclass
class ObservationTransform:
    original_size: Tuple[int, int]      # (w, h) of the environment screenshot
    processed_size: Tuple[int, int]     # (w, h) of the model-visible image
    scale: float                        # uniform scale processed/original
    resized: bool

    def to_model_space(self, x: float, y: float) -> Tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def to_env_space(self, x: float, y: float) -> Tuple[int, int]:
        w, h = self.original_size
        ex, ey = int(round(x / self.scale)), int(round(y / self.scale))
        return min(max(ex, 0), w - 1), min(max(ey, 0), h - 1)

    def as_log_dict(self) -> dict:
        return {
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
            "scale": self.scale,
            "resized": self.resized,
        }


def preprocess_screenshot(image: Image.Image, max_long_side: int = 1600,
                          interpolation=Image.LANCZOS) -> Tuple[Image.Image, ObservationTransform]:
    """Deterministic preprocessing pinned to the training policy: never
    upscale, preserve aspect, cap the long side."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    ow, oh = image.size
    long_side = max(ow, oh)
    if long_side <= max_long_side:
        return image, ObservationTransform((ow, oh), (ow, oh), 1.0, False)
    scale = max_long_side / float(long_side)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    if nh > oh or nw > ow:  # cannot happen, but never upscale defensively
        scale = 1.0
        return image, ObservationTransform((ow, oh), (ow, oh), 1.0, False)
    resized = image.resize((nw, nh), interpolation)
    return resized, ObservationTransform((ow, oh), (nw, nh), scale, True)


def load_screenshot(path_or_bytes) -> Tuple[Image.Image, ObservationTransform]:
    img = Image.open(path_or_bytes)
    return preprocess_screenshot(img)


def point_roundtrip(transform: ObservationTransform, x: int, y: int) -> Tuple[int, int]:
    """Env point -> model space -> back to env space (used by tests and
    logging; error stays sub-pixel for all sane resolutions)."""
    return transform.to_env_space(*transform.to_model_space(x, y))
