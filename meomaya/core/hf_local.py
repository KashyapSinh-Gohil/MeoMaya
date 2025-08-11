from __future__ import annotations

from typing import Any, Dict, List, Optional

from .hardware import select_device


class LocalTransformerClassifier:
    """
    Local-only text classifier using Hugging Face Transformers.

    Requirements:
      - transformers, torch installed (optional extra)
      - model_name_or_path must point to a local directory (no network)

    This class NEVER downloads models. It will error if the given model path
    isn't available locally.
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.device_str = device or select_device(prefer_gpu=True)
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError(
                "Install extras: pip install 'meomaya[hf]' to use LocalTransformerClassifier"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, local_files_only=True
        )
        self.device = self._resolve_device(self.device_str)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device_str: str):
        if device_str == "cuda":
            return self.torch.device("cuda")
        if device_str == "mps":
            return self.torch.device("mps")
        return self.torch.device("cpu")

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc)
            probs = out.logits.softmax(dim=-1).cpu().tolist()
        id2label = getattr(self.model.config, "id2label", {i: str(i) for i in range(len(probs[0]))})
        results: List[Dict[str, Any]] = []
        for p in probs:
            best_idx = int(max(range(len(p)), key=lambda i: p[i]))
            results.append({
                "label": id2label.get(best_idx, str(best_idx)),
                "scores": p,
            })
        return results


class LocalTransformerPOSTagger:
    """
    Local-only token classification (e.g., POS/NER) using HF Transformers.
    Requires a local model path. No downloads.
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.device_str = device or select_device(prefer_gpu=True)
        try:
            import torch  # type: ignore
            from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Install extras: pip install 'meomaya[hf]' to use LocalTransformerPOSTagger"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path, local_files_only=True
        )
        self.device = self._resolve_device(self.device_str)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device_str: str):
        if device_str == "cuda":
            return self.torch.device("cuda")
        if device_str == "mps":
            return self.torch.device("mps")
        return self.torch.device("cpu")

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        if not tokens:
            return []
        text = " ".join(tokens)
        enc = self.tokenizer(text, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc)
        logits = out.logits[0].cpu()
        ids = logits.argmax(-1).tolist()
        id2label = getattr(self.model.config, "id2label", {i: str(i) for i in range(logits.shape[-1])})
        # Map back tokens approximately (basic whitespace tokenization)
        word_pieces = self.tokenizer.convert_ids_to_tokens(self.tokenizer(text)["input_ids"])  # type: ignore
        # Heuristic: align original tokens to first wordpiece after basic split
        tagged: list[tuple[str, str]] = []
        idx = 1  # skip [CLS]
        for tok in tokens:
            while idx < len(word_pieces) and word_pieces[idx].startswith("##"):
                idx += 1
            label = id2label.get(ids[idx], "X") if idx < len(ids) else "X"
            tagged.append((tok, label))
            idx += 1
        return tagged


class LocalTransformerNERTagger:
    """
    Local-only NER tagger using token classification models.
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.device_str = device or select_device(prefer_gpu=True)
        try:
            import torch  # type: ignore
            from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Install extras: pip install 'meomaya[hf]' to use LocalTransformerNERTagger"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path, local_files_only=True
        )
        self.device = self._resolve_device(self.device_str)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device_str: str):
        if device_str == "cuda":
            return self.torch.device("cuda")
        if device_str == "mps":
            return self.torch.device("mps")
        return self.torch.device("cpu")

    def tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        if not tokens:
            return []
        text = " ".join(tokens)
        enc = self.tokenizer(text, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc)
        logits = out.logits[0].cpu()
        ids = logits.argmax(-1).tolist()
        id2label = getattr(self.model.config, "id2label", {i: str(i) for i in range(logits.shape[-1])})
        word_pieces = self.tokenizer.convert_ids_to_tokens(self.tokenizer(text)["input_ids"])  # type: ignore
        tagged: list[tuple[str, str]] = []
        idx = 1
        for tok in tokens:
            while idx < len(word_pieces) and word_pieces[idx].startswith("##"):
                idx += 1
            label = id2label.get(ids[idx], "O") if idx < len(ids) else "O"
            tagged.append((tok, label))
            idx += 1
        return tagged


class LocalTransformerParser:
    """
    Local-only seq2seq parser that outputs a bracketed parse string.
    Requires a local text2text model fine-tuned for parsing.
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.device_str = device or select_device(prefer_gpu=True)
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Install extras: pip install 'meomaya[hf]' to use LocalTransformerParser"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path, local_files_only=True
        )
        self.device = self._resolve_device(self.device_str)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device_str: str):
        if device_str == "cuda":
            return self.torch.device("cuda")
        if device_str == "mps":
            return self.torch.device("mps")
        return self.torch.device("cpu")

    def parse(self, text: str) -> str:
        enc = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=128)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


__all__ = [
    "LocalTransformerClassifier",
    "LocalTransformerPOSTagger",
    "LocalTransformerNERTagger",
    "LocalTransformerParser",
]


