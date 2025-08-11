# MeoMaya Documentation

Welcome to the official documentation for MeoMaya, a pure-Python, low-energy, high-speed NLP framework. This guide provides everything you need to know to use MeoMaya effectively.

---

## Table of Contents

1.  [Introduction](#introduction)
2.  [Installation](#installation)
3.  [Quick Start](#quick-start)
4.  [Core Components](#core-components)
    -   [Normalizer](#normalizer)
    -   [Tokenizer](#tokenizer)
    -   [Tagger](#tagger)
    -   [Parser](#parser)
5.  [Machine Learning Utilities](#machine-learning-utilities)
    -   [Vectorizer](#vectorizer)
    -   [Classifier](#classifier)
6.  [Command-Line Interface (CLI)](#command-line-interface-cli)
7.  [Advanced Usage](#advanced-usage)
    -   [Building a Custom Pipeline](#building-a-custom-pipeline)
8.  [API Reference](#api-reference)
    -   [Core Module](#core-module)
    -   [ML Module](#ml-module)
9.  [Troubleshooting](#troubleshooting)

---

## 1. Introduction

MeoMaya is designed for simplicity and performance, offering a complete pipeline for natural language processing tasks (normalize → tokenize → tag → parse) along with a lightweight, pure-Python machine learning stack. It's an excellent choice for developers and researchers who need an efficient and easy-to-understand NLP toolkit.

---

## 2. Installation

### Prerequisites

-   Python 3.11 or higher
-   `pip` package manager

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KashyapSinh-Gohil/meomaya.git
    cd meomaya
    ```

2.  **Create virtual environment and install runtime deps:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r meomaya/requirements.txt
    ```

3.  **Optional Dependencies:**
    -   For **API and development tools**, install:
        ```bash
        pip install -r requirements-dev.txt
        ```

---

## 3. Quick Start

### Using the `Modelify` Class

The easiest way to use MeoMaya is with the `Modelify` class, which runs the entire text processing pipeline.

```python
from meomaya.core.modelify import Modelify

# Initialize the model for text processing
m = Modelify(mode="text")

# Run the pipeline on your text
result = m.run("Hello world!")

# The result will be a dictionary containing the processed text
print(result)
```

### Using the CLI

You can also use MeoMaya directly from your terminal.

```bash
python -m meomaya "Hello world!" --mode text
```

---

## 4. Core Components

MeoMaya's core is a pipeline of four components for text processing.

### Normalizer

The `Normalizer` standardizes text by converting it to lowercase and applying other language-specific rules.

```python
from meomaya.core.normalizer import Normalizer

normalizer = Normalizer(lang="en")
normalized_text = normalizer.normalize("This is an EXAMPLE.")
# Output: "this is an example."
```

### Tokenizer

The `Tokenizer` splits a string of text into a list of tokens (words, punctuation, etc.).

```python
from meomaya.core.tokenizer import Tokenizer

tokenizer = Tokenizer(lang="en")
tokens = tokenizer.tokenize("Hello, world!")
# Output: ['Hello', ',', 'world', '!']
```

### Tagger

The `Tagger` assigns a Part-of-Speech (POS) tag to each token.

```python
from meomaya.core.tagger import Tagger

tagger = Tagger(lang="en")
tagged_tokens = tagger.tag(['hello', 'world'])
# Output: [('hello', 'UH'), ('world', 'NN')]
```

### Parser

The `Parser` creates a dependency parse tree from a list of tagged tokens.

```python
from meomaya.core.parser import Parser

parser = Parser(lang="en")
parse_tree = parser.parse([('hello', 'UH'), ('world', 'NN')])
# Output: {'tree': [('hello', 'UH'), ('world', 'NN')]}
```

---

## 5. Machine Learning Utilities

MeoMaya includes a pure-Python ML stack for basic machine learning tasks.

### Vectorizer

A `Vectorizer` that implements TF-IDF to convert a collection of raw documents to a matrix of TF-IDF features.

```python
from meomaya.ml.vectorizer import Vectorizer

texts = ["MeoMaya is great", "I love NLP"]
vectorizer = Vectorizer()
X = vectorizer.fit_transform(texts)
# X is now a list of TF-IDF vectors
```

### Classifier

A centroid-based `Classifier` that uses cosine similarity. It's simple, fast, and effective for many text classification tasks.

```python
from meomaya.ml.classifier import Classifier
from meomaya.ml.vectorizer import Vectorizer

# Sample data
texts = ["I love this product", "I hate this product", "This is great", "This is awful"]
labels = ["positive", "negative", "positive", "negative"]

# Vectorize the text
vectorizer = Vectorizer()
X = vectorizer.fit_transform(texts)

# Train the classifier
clf = Classifier()
clf.train(X, labels)

# Make predictions
new_texts = ["I love it", "This is really bad"]
X_new = vectorizer.transform(new_texts)
predictions = clf.classify(X_new)
print(predictions)
# Output: ['positive', 'negative']
```

---

## 6. Command-Line Interface (CLI)

MeoMaya provides a CLI for easy access to its components.

### Basic Usage

The preferred way is using the module entry point:

```bash
python -m meomaya "Your text here" --mode text
```

### Options

Options:

-   `--mode`: Override auto-detected mode: `text`, `audio`, `image`, `video`, `3d`, `fusion` (optional).
-   `--model`: Model name (placeholder; reserved for future model selection).

### Examples

-   **Run the complete pipeline:**
    ```bash
    python -m meomaya "Hello world!" --mode text
    ```

---

## 7. Advanced Usage

### Building a Custom Pipeline

You can easily create your own custom pipeline by combining the components.

```python
from meomaya.core.normalizer import Normalizer
from meomaya.core.tokenizer import Tokenizer
from meomaya.core.tagger import Tagger
from meomaya.core.parser import Parser

def custom_pipeline(text: str, lang: str = "en"):
    """A custom NLP pipeline."""
    normalizer = Normalizer(lang)
    tokenizer = Tokenizer(lang)
    tagger = Tagger(lang)
    parser = Parser(lang)

    normalized_text = normalizer.normalize(text)
    tokens = tokenizer.tokenize(normalized_text)
    tagged_tokens = tagger.tag(tokens)
    parse_tree = parser.parse(tagged_tokens)

    return {
        'normalized': normalized_text,
        'tokens': tokens,
        'tagged': tagged_tokens,
        'parsed': parse_tree,
    }

# Run the custom pipeline
result = custom_pipeline("This is a custom pipeline.")
print(result)
```

---

## 8. API Reference

This section provides a summary of the classes and methods in MeoMaya.

### Core Module

-   `Normalizer(lang: str = "en")`
    -   `normalize(text: str) -> str`
-   `Tokenizer(lang: str = "en")`
    -   `tokenize(text: str) -> list[str]`
-   `Tagger(lang: str = "en")`
    -   `tag(tokens: list[str]) -> list[tuple[str, str]]`
-   `Parser(lang: str = "en")`
    -   `parse(tagged_tokens: list[tuple[str, str]]) -> dict`

### ML Module

-   `Vectorizer()`
    -   `fit(documents: list[str])`
    -   `transform(documents: list[str]) -> list[list[float]]`
    -   `fit_transform(documents: list[str]) -> list[list[float]]`
-   `Classifier()`
    -   `train(X: list[list[float]], y: list[str])`
    -   `classify(X: list[list[float]]) -> list[str]`

---

## 9. Troubleshooting

### Common Issues

1.  **ImportError for `indic_nlp_library`**: If you are working with Indian languages and get an import error, make sure you have installed the optional dependency: `pip install indic-nlp-library`.
2.  **Incorrect Path for Corpus**: When using the CLI with a corpus file, ensure you provide the correct path to the file.
3.  **Performance**: For very large datasets, consider processing the data in batches to manage memory usage.

### Getting Help

If you're stuck, you can:
-   Review the test files in the `tests/` directory for more usage examples.
-   Open an issue on the [GitHub issue tracker](https://github.com/KashyapSinh-Gohil/meomaya/issues).

