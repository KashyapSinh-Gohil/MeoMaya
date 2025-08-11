import re
from typing import List


class Tokenizer:
    """
    A tokenizer that supports both English and Indian languages.
    """

    def __init__(self, lang: str = "en"):
        """
        Initializes the tokenizer.

        Args:
            lang (str, optional): The language to use for tokenization (e.g., "en", "hi"). Defaults to "en".
        """
        self.lang = lang
        # Regex for English tokenization
        self.en_token_pattern = re.compile(r"(\w+(?:-\w+)+|\w+|[.,!?;:()\"\\]|\S)")
        # Regex for simple Indian language tokenization
        self.in_token_pattern = re.compile(r"([\u0900-\u097F]+|[.,!?;:()\"']|\S)")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a string into a list of tokens.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        if not isinstance(text, str):
            raise TypeError("Input to tokenize() must be a string.")
        
        # Handle empty or whitespace-only strings
        if not text or not text.strip():
            return []

        if self.lang == "en":
            tokens = self.en_token_pattern.findall(text)
            # Filter out empty tokens
            return [token for token in tokens if token.strip()]
        else:
            # For Indian languages, a simple regex-based tokenizer
            tokens = self.in_token_pattern.findall(text)
            # Filter out empty tokens
            return [token for token in tokens if token.strip()]
