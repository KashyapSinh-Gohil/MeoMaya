import nltk
from typing import List, Tuple

class Parser:
    """
    A chart parser that uses a simple context-free grammar.
    """

    def __init__(self, lang: str = "en"):
        """
        Initializes the parser and defines a simple grammar.

        Args:
            lang (str, optional): The language to use for parsing. Defaults to "en".
        """
        self.lang = lang
        if self.lang == "en":
            self.grammar = nltk.CFG.fromstring("""
                S -> NP VP
                NP -> Det N | 'I'
                VP -> V NP | V
                Det -> 'a' | 'an' | 'the'
                N -> 'dog' | 'cat' | 'sentence'
                V -> 'saw' | 'is'
            """)
            self.parser = nltk.ChartParser(self.grammar)

    def parse(self, tagged_tokens: List[Tuple[str, str]]) -> dict:
        """
        Parses a list of tagged tokens and returns a parse tree.

        Args:
            tagged_tokens (List[Tuple[str, str]]): A list of (token, tag) tuples.

        Returns:
            dict: A dictionary representing the parse tree.
        """
        if self.lang == "en":
            # For English, use simple rule-based parsing since NLTK parser is too restrictive
            return self._simple_parse(tagged_tokens)
        else:
            # For other languages, use the simple rule-based parser
            parsed_elements = []
            i = 0
            while i < len(tagged_tokens):
                token, tag = tagged_tokens[i]

                # Rule 1: Noun Phrase (Adjective + Noun)
                if (
                    tag.startswith("J")
                    and i + 1 < len(tagged_tokens)
                    and tagged_tokens[i + 1][1].startswith("N")
                ):
                    parsed_elements.append(
                        {"type": "NP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                    )
                    i += 2
                # Rule 2: Noun Phrase (Noun + Noun)
                elif (
                    tag.startswith("N")
                    and i + 1 < len(tagged_tokens)
                    and tagged_tokens[i + 1][1].startswith("N")
                ):
                    parsed_elements.append(
                        {"type": "NP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                    )
                    i += 2
                # Rule 3: Verb Phrase (Verb + Noun Phrase - simplified)
                elif (
                    tag.startswith("V")
                    and i + 1 < len(tagged_tokens)
                    and tagged_tokens[i + 1][1].startswith("N")
                ):
                    # This is a very simple VP, ideally it would consume an NP
                    parsed_elements.append(
                        {"type": "VP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                    )
                    i += 2
                # Rule 4: Prepositional Phrase (Preposition + Noun Phrase - simplified)
                elif (
                    (tag.startswith("P") or tag == "IN")
                    and i + 1 < len(tagged_tokens)
                    and tagged_tokens[i + 1][1].startswith("N")
                ):
                    # This is a very simple PP, ideally it would consume an NP
                    parsed_elements.append(
                        {"type": "PP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                    )
                    i += 2
                else:
                    parsed_elements.append(tagged_tokens[i])
                    i += 1
            return {"tree": parsed_elements}
    
    def _simple_parse(self, tagged_tokens: List[Tuple[str, str]]) -> dict:
        """
        Simple rule-based parser that works for any tokens.
        """
        parsed_elements = []
        i = 0
        while i < len(tagged_tokens):
            token, tag = tagged_tokens[i]

            # Rule 1: Noun Phrase (Adjective + Noun)
            if (
                tag.startswith("J")
                and i + 1 < len(tagged_tokens)
                and tagged_tokens[i + 1][1].startswith("N")
            ):
                parsed_elements.append(
                    {"type": "NP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                )
                i += 2
            # Rule 2: Noun Phrase (Determiner + Noun)
            elif (
                tag in ["DT", "AT"]
                and i + 1 < len(tagged_tokens)
                and tagged_tokens[i + 1][1].startswith("N")
            ):
                parsed_elements.append(
                    {"type": "NP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                )
                i += 2
            # Rule 3: Verb Phrase (Verb + Noun)
            elif (
                tag.startswith("V")
                and i + 1 < len(tagged_tokens)
                and tagged_tokens[i + 1][1].startswith("N")
            ):
                parsed_elements.append(
                    {"type": "VP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                )
                i += 2
            # Rule 4: Prepositional Phrase (Preposition + Noun)
            elif (
                tag == "IN"
                and i + 1 < len(tagged_tokens)
                and tagged_tokens[i + 1][1].startswith("N")
            ):
                parsed_elements.append(
                    {"type": "PP", "children": [tagged_tokens[i], tagged_tokens[i + 1]]}
                )
                i += 2
            else:
                parsed_elements.append(tagged_tokens[i])
                i += 1
        return {"tree": parsed_elements}
