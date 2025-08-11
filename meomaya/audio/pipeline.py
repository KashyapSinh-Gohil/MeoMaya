from __future__ import annotations

from typing import Any, Dict

from meomaya.core.modelify import BasePipeline
from meomaya.core.utils import is_path_like


class AudioPipeline(BasePipeline):
    mode = "audio"

    def process(self, input_data: Any) -> Dict[str, Any]:
        path = str(input_data)
        meta: Dict[str, Any] = {"path": path if is_path_like(path) else None}
        # Try wave for simple WAV metadata
        try:
            import wave

            if is_path_like(path) and path.lower().endswith(".wav"):
                with wave.open(path, "rb") as wf:
                    meta.update({
                        "channels": wf.getnchannels(),
                        "sample_width": wf.getsampwidth(),
                        "frame_rate": wf.getframerate(),
                        "n_frames": wf.getnframes(),
                        "duration_sec": wf.getnframes() / max(1, wf.getframerate()),
                    })
        except Exception:
            pass
        # Try librosa for duration
        try:
            import librosa  # type: ignore

            if is_path_like(path):
                duration = float(librosa.get_duration(filename=path))
                meta.setdefault("duration_sec", duration)
        except Exception:
            pass
        return {"mode": self.mode, "results": meta}


__all__ = ["AudioPipeline"]


