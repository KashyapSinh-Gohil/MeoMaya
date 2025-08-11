from __future__ import annotations

from typing import Any, Dict

from meomaya.core.modelify import BasePipeline
from meomaya.core.utils import is_path_like


class ImagePipeline(BasePipeline):
    mode = "image"

    def process(self, input_data: Any) -> Dict[str, Any]:
        path = str(input_data)
        meta: Dict[str, Any] = {"path": path if is_path_like(path) else None}
        # Try PIL for metadata
        try:
            from PIL import Image  # type: ignore

            if is_path_like(path):
                with Image.open(path) as img:
                    meta.update({
                        "format": img.format,
                        "mode": img.mode,
                        "size": img.size,
                    })
        except Exception:
            pass
        # Try OpenCV for dimensions
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            if is_path_like(path):
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    meta.setdefault("size", (w, h))
                    meta.setdefault("channels", img.shape[2] if img.ndim == 3 else 1)
        except Exception:
            pass
        return {"mode": self.mode, "results": meta}


__all__ = ["ImagePipeline"]


