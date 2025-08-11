from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .utils import detect_mode_from_input


class BasePipeline:
    """Abstract base class for modality pipelines.

    All pipelines must implement a ``process`` method that returns a unified
    dictionary with at least ``mode`` and ``results`` keys.
    """

    mode: str = "base"

    def process(self, input_data: Any) -> Dict[str, Any]:
        raise NotImplementedError  # pragma: no cover


PipelineRegistry = Dict[str, Type[BasePipeline]]


class Modelify:
    """Central runner that dispatches to the appropriate modality pipeline.

    - Detects modality from file extension or explicit ``mode`` argument
    - Lazily imports and constructs the pipeline
    - Provides unified output format for all modalities
    - Plugin ready via ``register_plugin``
    """

    _registry: PipelineRegistry = {}

    def __init__(self, mode: Optional[str] = None, model: Optional[str] = None):
        self.explicit_mode = mode
        self.model = model or "default"

        # Register built-ins once
        if not Modelify._registry:
            self._register_builtins()

    # ---------------------- Public API ----------------------
    def run(self, input_data: Any, mode: Optional[str] = None) -> Dict[str, Any]:
        resolved_mode = mode or self.explicit_mode or detect_mode_from_input(input_data)
        if not resolved_mode:
            raise ValueError(
                "Unable to detect mode. Provide 'mode' explicitly or pass a supported file path."
            )

        pipeline_cls = self._get_pipeline_class(resolved_mode)
        pipeline = pipeline_cls()  # model selection hook could be used here
        output = pipeline.process(input_data)

        # Ensure unified envelope
        if "mode" not in output:
            output["mode"] = resolved_mode
        if "model" not in output:
            output["model"] = self.model
        if "meta" not in output:
            output["meta"] = {}
        return output

    @classmethod
    def register_plugin(cls, mode: str, pipeline_class: Type[BasePipeline]) -> None:
        mode_key = mode.lower().strip()
        if not mode_key:
            raise ValueError("mode cannot be empty")
        if not issubclass(pipeline_class, BasePipeline):
            raise TypeError("pipeline_class must inherit from BasePipeline")
        cls._registry[mode_key] = pipeline_class

    # ---------------------- Internals ----------------------
    def _get_pipeline_class(self, mode: str) -> Type[BasePipeline]:
        mode_key = mode.lower().strip()
        if mode_key not in Modelify._registry:
            raise KeyError(f"No pipeline registered for mode '{mode_key}'")
        return Modelify._registry[mode_key]

    def _register_builtins(self) -> None:
        # Local imports to keep base import light-weight
        from meomaya.text.pipeline import TextPipeline
        # Optional modalities (lightweight placeholders for now)
        try:
            from meomaya.image.pipeline import ImagePipeline  # type: ignore
        except Exception:  # pragma: no cover - optional
            ImagePipeline = None  # type: ignore
        try:
            from meomaya.audio.pipeline import AudioPipeline  # type: ignore
        except Exception:  # pragma: no cover - optional
            AudioPipeline = None  # type: ignore
        try:
            from meomaya.video.pipeline import VideoPipeline  # type: ignore
        except Exception:  # pragma: no cover - optional
            VideoPipeline = None  # type: ignore

        Modelify._registry = {
            "text": TextPipeline,
        }
        # Register optional ones if import worked
        if ImagePipeline is not None:
            Modelify._registry["image"] = ImagePipeline
        if AudioPipeline is not None:
            Modelify._registry["audio"] = AudioPipeline
        if VideoPipeline is not None:
            Modelify._registry["video"] = VideoPipeline


__all__ = ["Modelify", "BasePipeline"]
