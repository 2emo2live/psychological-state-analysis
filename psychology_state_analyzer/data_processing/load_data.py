from io import StringIO

import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path: str):
    file_content = dvc.api.read(path=file_path, remote="data_remote")

    df = pd.read_csv(StringIO(file_content))
    df = df.dropna()

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["status"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["statement"],
        df["label_encoded"],
        test_size=0.3,
        random_state=19,
        stratify=df["label_encoded"],
    )

    # Затем разделим temp на validation (15%) и test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=19, stratify=y_temp
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
