from __future__ import annotations

from typing import Any, Dict

from meomaya.core.modelify import BasePipeline
from meomaya.core.utils import is_path_like


class VideoPipeline(BasePipeline):
    mode = "video"

    def process(self, input_data: Any) -> Dict[str, Any]:
        path = str(input_data)
        meta: Dict[str, Any] = {"path": path if is_path_like(path) else None}
        # Try OpenCV for basic video metadata
        try:
            import cv2  # type: ignore

            if is_path_like(path):
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    duration = frames / fps if fps > 0 else 0.0
                    meta.update({
                        "size": (width, height),
                        "fps": fps,
                        "n_frames": frames,
                        "duration_sec": duration,
                    })
                cap.release()
        except Exception:
            pass
        return {"mode": self.mode, "results": meta}


__all__ = ["VideoPipeline"]


