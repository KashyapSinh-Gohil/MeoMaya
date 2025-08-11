import json
from typing import Dict, List


class CorpusLoader:
    """
    A corpus loader for the custom .meydata format.
    """

    def __init__(self):
        pass  # No longer needs corpus_path in init  # pragma: no cover

    @staticmethod
    def load_meydata(corpus_path: str) -> List[Dict]:
        """
        Loads a .meydata file and returns a list of annotated sentences.

        Args:
            corpus_path (str): The path to the .meydata file.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents an annotated sentence.
        """
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Error: Corpus file not found at {corpus_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in corpus file: {corpus_path}")
            return []
