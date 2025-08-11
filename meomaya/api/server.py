from __future__ import annotations

from typing import Any, List

from fastapi import FastAPI
from pydantic import BaseModel

from meomaya.core.modelify import Modelify


class RunRequest(BaseModel):
    input: Any
    mode: str | None = None


class BatchRunRequest(BaseModel):
    inputs: List[Any]
    mode: str | None = None


app = FastAPI(title="MeoMaya API", version="0.1.0")

# Cache pipelines/engine in app state to avoid per-request init overhead
class _State:
    engine: Modelify | None = None


state = _State()


@app.on_event("startup")
def _startup():
    # Lazy-initialize a default engine; mode can be overridden per request
    state.engine = Modelify()


@app.post("/run")
def run(req: RunRequest):
    engine = state.engine or Modelify()
    return engine.run(req.input, mode=req.mode)


@app.post("/run/batch")
def run_batch(req: BatchRunRequest):
    engine = state.engine or Modelify()
    if req.mode == "text":
        # Use batch API for text pipeline when possible
        pipeline = engine._get_pipeline_class("text")()
        return pipeline.process_batch(req.inputs)
    return [engine.run(item, mode=req.mode) for item in req.inputs]


