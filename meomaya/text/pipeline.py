from __future__ import annotations

from typing import Any, Dict

from meomaya.core import Normalizer, Parser, Tagger, Tokenizer
from meomaya.core.modelify import BasePipeline
from meomaya.core.utils import is_path_like


class TextPipeline(BasePipeline):
    mode = "text"

    def __init__(
        self,
        use_hf_classifier: bool = False,
        hf_model_path: str | None = None,
        use_hf_pos: bool = False,
        hf_pos_model_path: str | None = None,
        use_hf_parser: bool = False,
        hf_parser_model_path: str | None = None,
    ):
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer()
        self.tagger = Tagger()
        self.parser = Parser()
        self.use_hf_classifier = use_hf_classifier
        # Allow configuration via environment variable if not provided
        import os

        self.hf_model_path = hf_model_path or os.getenv("MEOMAYA_HF_CLS_PATH")
        self.use_hf_pos = use_hf_pos
        self.hf_pos_model_path = hf_pos_model_path or os.getenv("MEOMAYA_HF_POS_PATH")
        self.use_hf_parser = use_hf_parser
        self.hf_parser_model_path = hf_parser_model_path or os.getenv("MEOMAYA_HF_PARSER_PATH")

    def process(self, input_data: Any) -> Dict[str, Any]:
        if is_path_like(input_data):
            with open(input_data, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = str(input_data)

        normalized = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(normalized)
        if self.use_hf_pos and self.hf_pos_model_path:
            try:
                from meomaya.core.hf_local import LocalTransformerPOSTagger

                hf_pos = LocalTransformerPOSTagger(self.hf_pos_model_path)
                tags = hf_pos.tag(tokens)
            except Exception:
                tags = self.tagger.tag(tokens)
        else:
            tags = self.tagger.tag(tokens)
        if self.use_hf_parser and self.hf_parser_model_path:
            try:
                from meomaya.core.hf_local import LocalTransformerParser

                hf_parser = LocalTransformerParser(self.hf_parser_model_path)
                parse_str = hf_parser.parse(normalized)
                parsed = {"tree": parse_str}
            except Exception:
                parsed = self.parser.parse(tags)
        else:
            parsed = self.parser.parse(tags)

        # Minimal summarize/classify
        summary = " ".join(tokens[:20])
        classification = "neutral"
        if self.use_hf_classifier and self.hf_model_path:
            try:
                from meomaya.core.hf_local import LocalTransformerClassifier

                clf = LocalTransformerClassifier(self.hf_model_path)
                pred = clf.predict([text])
                if pred:
                    classification = pred[0].get("label", "neutral")
            except Exception:
                # Stay fully offline and optional; fall back silently
                classification = "neutral"

        return {
            "mode": self.mode,
            "results": {
                "normalized": normalized,
                "tokens": tokens,
                "tags": tags,
                "parse": parsed,
                "summary": summary,
                "classification": classification,
            },
        }

    def process_batch(self, inputs: list[Any]) -> list[Dict[str, Any]]:
        """Process a list of inputs sequentially.

        This is a simple reference batch API to enable higher throughput without
        changing the single-item semantics. It avoids external services.
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list")
        return [self.process(item) for item in inputs]

    def process_stream(self, inputs: Any):
        """Process a stream (iterator/generator) lazily.

        Yields one output per input to enable streaming pipelines.
        """
        for item in inputs:
            yield self.process(item)


__all__ = ["TextPipeline"]
