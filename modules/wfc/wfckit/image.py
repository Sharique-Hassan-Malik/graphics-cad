"""A minimal PNG writer, so the only dependency is NumPy.

A PNG is a signature, then a sequence of length-tagged, CRC-checked chunks: IHDR
(dimensions and colour type), IDAT (the pixels, filtered per scanline then
zlib-compressed), and IEND. This writes 8-bit RGB with the "None" per-row filter
— the simplest valid encoding, and plenty for flat tile art.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png(path: str, rgb: np.ndarray) -> str:
    """Write an (H, W, 3) uint8 array as an RGB PNG."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB array")
    h, w, _ = rgb.shape
    # Each scanline is prefixed with a filter-type byte (0 = None).
    raw = np.hstack([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)]).tobytes()
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))  # 8-bit RGB
    png += _chunk(b"IDAT", zlib.compress(raw, 9))
    png += _chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(png)
    return path
