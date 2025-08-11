from __future__ import annotations

import os
from typing import Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None  # type: ignore


class ModelRegistry:
    """Minimal local model registry using a YAML file.

    Looks for MEOMAYA_MODEL_REGISTRY env var; otherwise uses ./meomaya/models/registry.yaml if present.
    """

    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or os.getenv("MEOMAYA_MODEL_REGISTRY")
        if not self.registry_path:
            # Relative to project if exists
            here = os.path.dirname(os.path.dirname(__file__))
            default_path = os.path.join(here, "models", "registry.yaml")
            self.registry_path = default_path if os.path.exists(default_path) else ""
        self._data = {}
        if self.registry_path and yaml is not None and os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._data = yaml.safe_load(f) or {}
            except Exception:
                self._data = {}

    def get_path(self, key: str) -> Optional[str]:
        if not key:
            return None
        return self._data.get(key)


__all__ = ["ModelRegistry"]


