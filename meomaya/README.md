# MeoMaya

[![License](https://img.shields.io/badge/License-Polyform%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)

**A pure-Python, high-speed NLP framework with a clean, modular core for text processing. Now with optional multimodal pipelines, local-only transformers, a REST API, and hardware-aware execution (CPU/CUDA/MPS).**

MeoMaya is designed for simplicity, performance, and low resource consumption. It offers a complete and extensible pipeline for common natural language processing tasks (normalize → tokenize → tag → parse) along with a lightweight, pure-Python machine learning stack. It's an excellent choice for developers and researchers who need an efficient and easy-to-understand NLP toolkit without heavy dependencies.

---

## 🌟 Key Features

- **Complete NLP Pipeline**: Out-of-the-box normalization, tokenization, POS tagging, and parsing.
- **Pure-Python ML Stack**: Includes a TF-IDF vectorizer and a centroid-based classifier.
- **Optional Local Transformers**: Local-only Hugging Face wrapper (no network) via `meomaya[hf]` extra.
- **Multimodal Pipelines (Optional)**: `image`, `audio`, and `video` pipelines extract real metadata locally when PIL/OpenCV/librosa are available.
- **REST API (Optional)**: FastAPI server with `/run` and `/run/batch` endpoints via `meomaya[api]`.
- **Hardware-Aware**: Automatic device selection (CPU/CUDA/MPS) without forcing torch.
- **Lightweight and Fast**: Optimized for speed and low memory usage.
- **Indian Language Support**: Optional support for Indian languages via `indic-nlp-library`.
- **Command-Line Interface**: A simple and powerful CLI for quick text processing.
- **Extensible**: A modular design that is easy to build upon.

---

## 🚀 Getting Started

### Installation

For development, clone the repository and install runtime deps:

```bash
git clone https://github.com/KashyapSinh-Gohil/meomaya.git
cd meomaya
python -m venv .venv
source .venv/bin/activate
pip install -r meomaya/requirements.txt
```

Optional dev/API tools:

```bash
pip install -r requirements-dev.txt
```

### Quick Start: The NLP Pipeline

See the core MeoMaya pipeline in action. The library is built around a series of steps that process raw text into structured data.

```python
import pprint
from meomaya.core.normalizer import Normalizer
from meomaya.core.tokenizer import Tokenizer
from meomaya.core.tagger import Tagger
from meomaya.core.parser import Parser

# 1. Sample Text
text = "MeoMaya is a fast and lightweight NLP library!"

# 2. Normalize the text (lowercase, remove punctuation, etc.)
normalizer = Normalizer()
normalized_text = normalizer.normalize(text)
# Output: 'meomaya is a fast and lightweight nlp library'

# 3. Tokenize the text into words
tokenizer = Tokenizer()
tokens = tokenizer.tokenize(normalized_text)
# Output: ['meomaya', 'is', 'a', 'fast', 'and', 'lightweight', 'nlp', 'library']

# 4. Tag the tokens with their Part-of-Speech (POS)
tagger = Tagger()
tagged_tokens = tagger.tag(tokens)
# Output: [('meomaya', 'NN'), ('is', 'VBZ'), ..., ('library', 'NN')]

# 5. Parse the tagged tokens to create a syntax tree
parser = Parser()
parsed_tree = parser.parse(tagged_tokens)

# Print the final parsed structure
pprint.pprint(parsed_tree)
```

You can also perform quick analyses directly from the command line:

```bash
python -m meomaya "Hello world!" --mode text

### REST API (optional)

Run a local API server (no third-party services):

```bash
uvicorn meomaya.api.server:app --host 0.0.0.0 --port 8000
```

Call it:

```bash
curl -X POST http://localhost:8000/run -H 'Content-Type: application/json' \
  -d '{"input": "Hello from MeoMaya!", "mode": "text"}'
```

### Optional HF in TextPipeline (offline)

```python
from meomaya.text.pipeline import TextPipeline

pipeline = TextPipeline(
    use_hf_classifier=True,
    hf_model_path="/path/to/local/text-classification-model",
    use_hf_pos=True,
    hf_pos_model_path="/path/to/local/pos-model",
)
print(pipeline.process("MeoMaya makes NLP easy!"))
```

### Batch API (text)

```bash
curl -X POST http://localhost:8000/run/batch -H 'Content-Type: application/json' \
  -d '{"inputs": ["hi", "there"], "mode": "text"}'
```

---

## 📚 Documentation

For a more detailed guide, including API references, advanced usage, and tutorials, please see our full documentation:

- **[MeoMaya Documentation](./docs/index.md)**

---

## 🧪 Testing

To ensure everything is set up correctly, you can run the consolidated test suite:

```bash
PYTHONPATH=. python -m pytest meomaya/tests -v
```

### Offline mode

Set `MEOMAYA_STRICT_OFFLINE=1` to avoid any NLTK downloads. In strict offline mode, the `Normalizer` and `Tagger` use lightweight built-in fallbacks.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes and add corresponding tests.
4.  Ensure all tests pass.
5.  Submit a pull request.

---

## 📄 License

This project is licensed under the **Polyform Noncommercial 1.0.0 License**. This means it is free for open-source and non-commercial use.

For commercial use, please contact the author, Kashyapsinh Gohil (kagohil000@gmail.com), to obtain a commercial license.

---

## 📞 Support

If you encounter any issues or have questions, please open an issue on the [GitHub issue tracker](https://github.com/KashyapSinh-Gohil/meomaya/issues).
