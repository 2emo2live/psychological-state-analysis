import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class BaselineModel:
    def __init__(
        self,
        tfidf_params: dict = None,
        logreg_params: dict = None,
        random_state: int = 42,
    ):
        self.tfidf_params = tfidf_params or {}
        self.logreg_params = logreg_params or {}
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.classifier = LogisticRegression(
            random_state=self.random_state, **self.logreg_params
        )
        self.label_encoder = LabelEncoder()

        self.is_fitted = False

    def fit(self, X_texts: pd.Series, y: pd.Series) -> "BaselineModel":
        y_encoded = self.label_encoder.fit_transform(y)
        X_tfidf = self.vectorizer.fit_transform(X_texts)
        self.classifier.fit(X_tfidf, y_encoded)
        self.is_fitted = True
        return self

    def predict(self, X_texts: pd.Series) -> np.ndarray:
        X_tfidf = self.vectorizer.transform(X_texts)
        preds = self.classifier.predict(X_tfidf)
        return preds

    def predict_proba(self, X_texts: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена.")
        X_tfidf = self.vectorizer.transform(X_texts)
        return self.classifier.predict_proba(X_tfidf)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.classifier, f)
        with open(path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(path / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        with open(path / "params.pkl", "wb") as f:
            pickle.dump(
                {
                    "tfidf_params": self.tfidf_params,
                    "logreg_params": self.logreg_params,
                    "random_state": self.random_state,
                },
                f,
            )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaselineModel":
        path = Path(path)
        with open(path / "model.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open(path / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(path / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        with open(path / "params.pkl", "rb") as f:
            params = pickle.load(f)

        model = cls(
            tfidf_params=params["tfidf_params"],
            logreg_params=params["logreg_params"],
            random_state=params["random_state"],
        )
        model.vectorizer = vectorizer
        model.classifier = classifier
        model.label_encoder = label_encoder
        model.is_fitted = True
        return model
