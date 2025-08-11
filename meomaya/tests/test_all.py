import pytest

# Core imports
from meomaya.core.normalizer import Normalizer
from meomaya.core.tokenizer import Tokenizer
from meomaya.core.tagger import Tagger
from meomaya.core.parser import Parser

# ML imports
from meomaya.ml.vectorizer import Vectorizer
from meomaya.ml.classifier import Classifier, _cosine_similarity


# ---------------- Normalizer Tests ----------------

@pytest.fixture
def normalizer():
    return Normalizer(lang="en")


def test_lowercase(normalizer):
    text = "This is a SAMPLE sentence."
    normalized_text = normalizer.normalize(text, options=["lowercase"])
    assert normalized_text == "this is a sample sentence."


def test_remove_punctuation(normalizer):
    text = "This is a sentence with punctuation!@#$%^&*()"
    normalized_text = normalizer.normalize(text, options=["remove_punctuation"])
    assert normalized_text == "This is a sentence with punctuation"


def test_strip(normalizer):
    text = "  This is a sentence with leading and trailing whitespace.  "
    normalized_text = normalizer.normalize(text, options=["strip"])
    assert normalized_text == "This is a sentence with leading and trailing whitespace."


def test_stem(normalizer):
    text = "running runs ran"
    normalized_text = normalizer.normalize(text, options=["stem"])
    assert normalized_text == "run run ran"


def test_lemmatize(normalizer):
    text = "running runs ran"
    normalized_text = normalizer.normalize(text, options=["lemmatize"])
    assert normalized_text == "running run ran"


def test_remove_stopwords(normalizer):
    text = "this is a sentence with some stopwords"
    normalized_text = normalizer.normalize(text, options=["remove_stopwords"])
    assert normalized_text == "sentence stopwords"


# ---------------- Tokenizer Tests ----------------

@pytest.fixture
def tokenizer():
    return Tokenizer(lang="en")


def test_tokenizer_simple_sentence(tokenizer):
    text = "hello world this is a test"
    expected = ["hello", "world", "this", "is", "a", "test"]
    assert tokenizer.tokenize(text) == expected


def test_tokenizer_with_punctuation(tokenizer):
    text = "hello, world! this is a test."
    expected = ["hello", ",", "world", "!", "this", "is", "a", "test", "."]
    assert tokenizer.tokenize(text) == expected


def test_tokenizer_hyphenated_words(tokenizer):
    text = "this is a state-of-the-art tokenizer"
    expected = ["this", "is", "a", "state-of-the-art", "tokenizer"]
    assert tokenizer.tokenize(text) == expected


def test_tokenizer_empty_string(tokenizer):
    assert tokenizer.tokenize("") == []


def test_tokenizer_whitespace_string(tokenizer):
    assert tokenizer.tokenize("   ") == []


# ---------------- Tagger Tests ----------------

@pytest.fixture
def tagger():
    return Tagger(lang="en")


def test_tagger_simple_sentence(tagger):
    tokens = ["This", "is", "a", "simple", "sentence", "."]
    tagged_tokens = tagger.tag(tokens)
    assert tagged_tokens == [("This", "DT"), ("is", "BEZ"), ("a", "AT"), ("simple", "JJ"), ("sentence", "NN"), (".", ".")]


def test_tagger_unknown_words(tagger):
    tokens = ["This", "is", "a", "sentence", "with", "some", "unknown", "words", "."]
    tagged_tokens = tagger.tag(tokens)
    assert len(tagged_tokens) == len(tokens)
    assert all(isinstance(token, str) and isinstance(tag, str) for token, tag in tagged_tokens)
    assert tagged_tokens[0][0] == "This"
    assert tagged_tokens[-1][0] == "."


# ---------------- Parser Tests ----------------

@pytest.fixture
def parser():
    return Parser(lang="en")


def test_parser_simple_sentence(parser):
    tagged_tokens = [("the", "DT"), ("cat", "NN"), ("saw", "VBD"), ("a", "DT"), ("dog", "NN")]
    tree = parser.parse(tagged_tokens)
    assert "tree" in tree
    assert len(tree["tree"]) > 0


def test_parser_ungrammatical_sentence(parser):
    tagged_tokens = [("the", "DT"), ("cat", "NN"), ("saw", "VBD"), ("a", "DT"), ("dog", "NN"), ("barked", "VBD")]
    tree = parser.parse(tagged_tokens)
    assert "tree" in tree
    assert len(tree["tree"]) > 0


# ---------------- ML: Vectorizer Tests ----------------

@pytest.fixture
def vectorizer():
    return Vectorizer()


def test_vectorizer_fit_empty(vectorizer):
    vectorizer.fit([])
    assert vectorizer.vocab == {}
    assert vectorizer.idf == {}
    assert not vectorizer._fitted


def test_vectorizer_fit_single_text(vectorizer):
    texts = ["hello world"]
    vectorizer.fit(texts)
    assert vectorizer.vocab == {"hello": 0, "world": 1}
    assert len(vectorizer.idf) == 2
    assert vectorizer._fitted


def test_vectorizer_fit_multiple_texts(vectorizer):
    texts = ["hello world", "foo bar hello"]
    vectorizer.fit(texts)
    assert vectorizer.vocab == {"bar": 0, "foo": 1, "hello": 2, "world": 3}
    assert len(vectorizer.idf) == 4
    assert vectorizer._fitted


def test_vectorizer_transform_unfitted(vectorizer):
    with pytest.raises(ValueError):
        vectorizer.transform(["hello"])


def test_vectorizer_transform_single_text(vectorizer):
    texts = ["hello world"]
    vectorizer.fit(texts)
    vectors = vectorizer.transform(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 2


def test_vectorizer_transform_multiple_texts(vectorizer):
    texts = ["hello world", "foo bar hello"]
    vectorizer.fit(texts)
    vectors = vectorizer.transform(["hello world", "foo bar hello"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 4
    assert len(vectors[1]) == 4


def test_vectorizer_fit_transform(vectorizer):
    texts = ["hello world", "foo bar hello"]
    vectors = vectorizer.fit_transform(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 4
    assert vectorizer._fitted


def test_vectorizer_vectorize(vectorizer):
    texts = ["hello world", "foo bar hello"]
    vectors = vectorizer.vectorize(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 4
    assert vectorizer._fitted


# ---------------- ML: Classifier Tests ----------------

@pytest.fixture
def classifier():
    return Classifier()


def test_classifier_train_empty(classifier):
    with pytest.raises(ValueError):
        classifier.train([], [])


def test_classifier_train_single_class(classifier):
    X = [[1, 2], [1.1, 2.1]]
    y = ["A", "A"]
    classifier.train(X, y)
    assert "A" in classifier.class_to_centroid
    assert classifier._fitted


def test_classifier_train_multiple_classes(classifier):
    X = [[1, 2], [1.1, 2.1], [10, 11], [10.1, 11.1]]
    y = ["A", "A", "B", "B"]
    classifier.train(X, y)
    assert "A" in classifier.class_to_centroid
    assert "B" in classifier.class_to_centroid
    assert classifier._fitted


def test_classifier_classify_untrained(classifier):
    with pytest.raises(ValueError):
        classifier.classify([[1, 2]])


def test_classifier_classify_single_vector(classifier):
    X_train = [[1, 2], [1.1, 2.1], [10, 11], [10.1, 11.1]]
    y_train = ["A", "A", "B", "B"]
    classifier.train(X_train, y_train)
    preds = classifier.classify([[1.05, 2.05]])
    assert preds == ["A"]


def test_classifier_classify_multiple_vectors(classifier):
    X_train = [[1, 2], [1.1, 2.1], [10, 11], [10.1, 11.1]]
    y_train = ["A", "A", "B", "B"]
    classifier.train(X_train, y_train)
    preds = classifier.classify([[1.05, 2.05], [10.05, 11.05]])
    assert preds == ["A", "B"]


def test_cosine_similarity():
    assert _cosine_similarity([1, 0], [0, 1]) == 0.0
    assert _cosine_similarity([1, 1], [1, 1]) == 1.0
    assert _cosine_similarity([1, 0], [1, 0]) == 1.0
    assert _cosine_similarity([1, 1], [-1, -1]) == -1.0
    assert _cosine_similarity([0, 0], [1, 1]) == 0.0


