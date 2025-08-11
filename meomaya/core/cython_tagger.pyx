
from typing import List, Tuple

# This is a Cython implementation of the Tagger class.
# To compile it, you'll need to add a build step to your pyproject.toml
# and install cython.

cdef class CythonTagger:
    """
    A simple rule-based POS tagger, optimized with Cython.
    """
    cdef dict tag_dict

    def __init__(self, lang: str = "en"):
        """
        Initializes the tagger.

        Args:
            lang (str, optional): The language to use for tagging. Defaults to "en".
        """
        self.tag_dict = {
            # Nouns
            "world": "NN", "sentence": "NN", "cat": "NN", "dog": "NN", "car": "NN",
            # Verbs
            "is": "VBZ", "are": "VBP", "was": "VBD", "were": "VBD", "run": "VB", "runs": "VBZ",
            # Adjectives
            "simple": "JJ", "big": "JJ", "small": "JJ", "red": "JJ", "good": "JJ",
            # Adverbs
            "quickly": "RB", "slowly": "RB", "very": "RB", "well": "RB",
            # Pronouns
            "i": "PRP", "you": "PRP", "he": "PRP", "she": "PRP", "it": "PRP", "we": "PRP", "they": "PRP",
            # Determiners
            "a": "DT", "an": "DT", "the": "DT", "this": "DT", "that": "DT",
            # Prepositions
            "in": "IN", "on": "IN", "at": "IN", "with": "IN", "for": "IN",
            # Conjunctions
            "and": "CC", "but": "CC", "or": "CC",
            # Punctuation
            ".": ".", ",": ",", "!": ".", "?": ".",
            # Interjections
            "hello": "UH", "oh": "UH",
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
        tagged_tokens = []
        for token in tokens:
            tag = self.tag_dict.get(token.lower(), "NN")  # Default to Noun
            tagged_tokens.append((token, tag))
        return tagged_tokens
