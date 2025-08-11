from __future__ import annotations

import os
from typing import Optional

EXT_TO_MODE = {
    # text
    ".txt": "text",
    ".md": "text",
    ".json": "text",
    # audio
    ".wav": "audio",
    ".mp3": "audio",
    ".m4a": "audio",
    # image
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".bmp": "image",
    # video
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    # 3D
    ".obj": "3d",
    ".ply": "3d",
}


def is_path_like(value) -> bool:
    return isinstance(value, str) and (os.path.exists(value) or _looks_like_path(value))


def _looks_like_path(value: str) -> bool:
    return any(sep in value for sep in ("/", "\\")) and "." in os.path.basename(
        value
    )  # pragma: no cover


def detect_mode_from_input(input_data) -> Optional[str]:
    if is_path_like(input_data):
        _, ext = os.path.splitext(str(input_data))
        return EXT_TO_MODE.get(ext.lower())
    # Default to text if it's a plain string
    if isinstance(input_data, str):
        return "text"
    return None


__all__ = ["detect_mode_from_input", "is_path_like", "EXT_TO_MODE"]
