import os
import nltk
from typing import List, Tuple

class Tagger:
    """
    A trigram tagger with backoff to bigram, unigram, and default taggers.
    """

    def __init__(self, lang: str = "en"):
        """
        Initializes the tagger and trains it on the Brown Corpus.

        Args:
            lang (str, optional): The language to use for tagging. Defaults to "en".
        """
        self.lang = lang
        if self.lang == "en":
            # Allow strict offline mode to avoid downloads
            strict_offline = os.getenv("MEOMAYA_STRICT_OFFLINE", "0") in {"1", "true", "True"}
            has_brown = True
            try:
                nltk.data.find('corpora/brown')
            except LookupError:
                has_brown = False

            if has_brown:
                train_sents = nltk.corpus.brown.tagged_sents()
                t0 = nltk.DefaultTagger('NN')
                t1 = nltk.UnigramTagger(train_sents, backoff=t0)
                t2 = nltk.BigramTagger(train_sents, backoff=t1)
                self.tagger = nltk.TrigramTagger(train_sents, backoff=t2)
            elif not strict_offline:
                # Attempt to download if allowed
                nltk.download('brown')
                train_sents = nltk.corpus.brown.tagged_sents()
                t0 = nltk.DefaultTagger('NN')
                t1 = nltk.UnigramTagger(train_sents, backoff=t0)
                t2 = nltk.BigramTagger(train_sents, backoff=t1)
                self.tagger = nltk.TrigramTagger(train_sents, backoff=t2)
            else:
                # Strict offline fallback: use lightweight dictionary-based tagger
                self.tagger = None
                self.tag_dict = {
                    # Nouns
                    "world": "NN",
                    "sentence": "NN",
                    "cat": "NN",
                    "dog": "NN",
                    "car": "NN",
                    # Verbs
                    "is": "VBZ",
                    "are": "VBP",
                    "was": "VBD",
                    "were": "VBD",
                    "run": "VB",
                    "runs": "VBZ",
                    # Adjectives
                    "simple": "JJ",
                    "big": "JJ",
                    "small": "JJ",
                    "red": "JJ",
                    "good": "JJ",
                    # Adverbs
                    "quickly": "RB",
                    "slowly": "RB",
                    "very": "RB",
                    "well": "RB",
                    # Pronouns
                    "i": "PRP",
                    "you": "PRP",
                    "he": "PRP",
                    "she": "PRP",
                    "it": "PRP",
                    "we": "PRP",
                    "they": "PRP",
                    # Determiners
                    "a": "DT",
                    "an": "DT",
                    "the": "DT",
                    "this": "DT",
                    "that": "DT",
                    # Prepositions
                    "in": "IN",
                    "on": "IN",
                    "at": "IN",
                    "with": "IN",
                    "for": "IN",
                    # Conjunctions
                    "and": "CC",
                    "but": "CC",
                    "or": "CC",
                    # Punctuation
                    ".": ".",
                    ",": ",",
                    "!": ".",
                    "?": ".",
                    # Interjections
                    "hello": "UH",
                    "oh": "UH",
                    # Other
                    "how": "WRB",
                }
        else:
            # For other languages, use the simple dictionary-based tagger
            self.tag_dict = {
                # Nouns
                "world": "NN",
                "sentence": "NN",
                "cat": "NN",
                "dog": "NN",
                "car": "NN",
                # Verbs
                "is": "VBZ",
                "are": "VBP",
                "was": "VBD",
                "were": "VBD",
                "run": "VB",
                "runs": "VBZ",
                # Adjectives
                "simple": "JJ",
                "big": "JJ",
                "small": "JJ",
                "red": "JJ",
                "good": "JJ",
                # Adverbs
                "quickly": "RB",
                "slowly": "RB",
                "very": "RB",
                "well": "RB",
                # Pronouns
                "i": "PRP",
                "you": "PRP",
                "he": "PRP",
                "she": "PRP",
                "it": "PRP",
                "we": "PRP",
                "they": "PRP",
                # Determiners
                "a": "DT",
                "an": "DT",
                "the": "DT",
                "this": "DT",
                "that": "DT",
                # Prepositions
                "in": "IN",
                "on": "IN",
                "at": "IN",
                "with": "IN",
                "for": "IN",
                # Conjunctions
                "and": "CC",
                "but": "CC",
                "or": "CC",
                # Punctuation
                ".": ".",
                ",": ",",
                "!": ".",
                "?": ".",
                # Interjections
                "hello": "UH",
                "oh": "UH",
                # Other
                "how": "WRB",
            }

    def tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Tags a list of tokens with their POS tags.

        Args:
            tokens (List[str]): A list of tokens.

        Returns:
            List[Tuple[str, str]]: A list of (token, tag) tuples.
        """
        if not isinstance(tokens, list) or not all(
            isinstance(t, str) for t in tokens
        ):
            raise TypeError("Input to tag() must be a list of strings.")
        if self.lang == "en" and getattr(self, "tagger", None) is not None:
            return self.tagger.tag(tokens)
        else:
            tagged_tokens = []
            for token in tokens:
                tag = self.tag_dict.get(token.lower(), "NN")  # Default to Noun
                tagged_tokens.append((token, tag))
            return tagged_tokens
