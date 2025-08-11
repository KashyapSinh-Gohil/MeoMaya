import os
import string
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

class Normalizer:
    """
    A text normalizer that supports both English and Indian languages.
    """

    def __init__(self, lang: str = "en"):
        """
        Initializes the normalizer.

        Args:
            lang (str, optional): The language to use for normalization (e.g., "en", "hi"). Defaults to "en".
        """
        self.lang = lang
        if self.lang == "en":
            strict_offline = os.getenv("MEOMAYA_STRICT_OFFLINE", "0") in {"1", "true", "True"}
            has_stop = True
            has_wordnet = True
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                has_stop = False
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                has_wordnet = False
            # Initialize tools with fallbacks
            self.stemmer = PorterStemmer()
            self.lemmatizer = None
            self.stopwords = set()
            if has_stop:
                self.stopwords = set(stopwords.words('english'))
            elif not strict_offline:
                nltk.download('stopwords')
                self.stopwords = set(stopwords.words('english'))
            # WordNet lemmatizer requires wordnet resource
            if has_wordnet:
                self.lemmatizer = WordNetLemmatizer()
            elif not strict_offline:
                nltk.download('wordnet')
                self.lemmatizer = WordNetLemmatizer()

    def normalize(
        self, text: str, options: list = ["lowercase", "remove_punctuation", "strip"]
    ) -> str:
        """
        Normalizes a string by applying a list of options.

        Args:
            text (str): The text to normalize.
            options (list, optional): A list of normalization options to apply. 
                Defaults to ["lowercase", "remove_punctuation", "strip"].
                Available options: "lowercase", "remove_punctuation", "strip", "stem", "lemmatize", "remove_stopwords".

        Returns:
            str: The normalized text.
        """
        if not isinstance(text, str):
            raise TypeError("Input to normalize() must be a string.")

        if self.lang == "en":
            if "lowercase" in options:
                text = text.lower()
            if "remove_punctuation" in options:
                text = text.translate(str.maketrans("", "", string.punctuation))
            if "remove_stopwords" in options:
                words = text.split()
                words = [word for word in words if word not in self.stopwords]
                text = " ".join(words)
            if "stem" in options:
                words = text.split()
                words = [self.stemmer.stem(word) for word in words]
                text = " ".join(words)
            if "lemmatize" in options and self.lemmatizer is not None:
                words = text.split()
                words = [self.lemmatizer.lemmatize(word) for word in words]
                text = " ".join(words)
            if "strip" in options:
                text = text.strip()
            return text
        else:
            # For Indian languages, a simple normalization
            return text.strip()