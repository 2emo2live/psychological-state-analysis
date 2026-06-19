from io import StringIO
from pathlib import Path

import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(
    root_path: str,
    dvc_path: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 19,
):
    file_content = dvc.api.read(path=dvc_path, remote="data_remote")
    df = pd.read_csv(StringIO(file_content))
    df = df.dropna()

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["status"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["statement"],
        df["label_encoded"],
        train_size=train_size,
        random_state=seed,
        stratify=df["label_encoded"],
    )

    relative_val_size = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=relative_val_size, random_state=seed, stratify=y_temp
    )

    output_path = Path(root_path)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame({"statement": X_train, "label": y_train})
    val_df = pd.DataFrame({"statement": X_val, "label": y_val})
    test_df = pd.DataFrame({"statement": X_test, "label": y_test})

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
