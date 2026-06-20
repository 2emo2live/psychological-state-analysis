import re

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab", quiet=True)


def preprocess_text(
    text: str, lowercase: bool, remove_urls: bool, remove_punct: bool, stem: bool
) -> str:
    if not isinstance(text, str):
        return ""

    if remove_urls:
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    if lowercase:
        text = text.lower()

    if remove_punct:
        text = re.sub(r"[^a-zA-Z\s]", "", text)

    if stem:
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        return " ".join(tokens)
    else:
        return text
