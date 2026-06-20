from functools import partial
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from psychology_state_analyzer.data_processing.baseline_preprocess import (
    preprocess_text,
)
from psychology_state_analyzer.model_class.baseline import BaselineModel


def macro_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        accs.append(accuracy_score(y_true[mask], y_pred[mask]))
    return np.mean(accs) if accs else 0.0


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    data_dir = Path(cfg.data.root_path)
    required_files = ["train.csv", "val.csv", "test.csv"]
    if not all((data_dir / f).exists() for f in required_files):
        load_data(
            root_path=cfg.data.root_path,
            dvc_path=cfg.data.dvc_path,
            train_size=cfg.data.train_split,
            val_size=cfg.data.val_split,
            seed=cfg.train.seed,
        )

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train_df["statement"], train_df["label"]
    X_val, y_val = val_df["statement"], val_df["label"]
    X_test, y_test = test_df["statement"], test_df["label"]

    lowercase = cfg.baseline.preprocessing.lowercase
    remove_urls = cfg.baseline.preprocessing.remove_urls
    remove_punct = cfg.baseline.preprocessing.remove_punct
    stem = cfg.baseline.preprocessing.stem

    preprocess = partial(
        preprocess_text,
        lowercase=lowercase,
        remove_urls=remove_urls,
        remove_punct=remove_punct,
        stem=stem,
    )

    X_train_prep = X_train.apply(preprocess)
    X_val_prep = X_val.apply(preprocess)
    X_test_prep = X_test.apply(preprocess)

    model = BaselineModel(
        tfidf_params={
            "max_features": cfg.baseline.tfidf.max_features,
            "ngram_range": tuple(cfg.baseline.tfidf.ngram_range),
            "min_df": cfg.baseline.tfidf.min_df,
            "max_df": cfg.baseline.tfidf.max_df,
        },
        logreg_params={
            "C": cfg.baseline.logreg.C,
            "max_iter": cfg.baseline.logreg.max_iter,
            "solver": cfg.baseline.logreg.solver,
            "multi_class": cfg.baseline.logreg.multi_class,
        },
        random_state=cfg.train.seed,
    )
    model.fit(X_train_prep, y_train)

    y_val_pred = model.predict(X_val_prep)
    y_test_pred = model.predict(X_test_prep)

    val_precision = precision_score(y_val, y_val_pred, average="macro", zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average="macro", zero_division=0)
    val_macro_acc = macro_accuracy(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
    val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted")

    test_precision = precision_score(
        y_test, y_test_pred, average="macro", zero_division=0
    )
    test_recall = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_macro_acc = macro_accuracy(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

    print("Validation metrics:")
    print(f"  Macro Precision: {val_precision:.4f}")
    print(f"  Macro Recall:    {val_recall:.4f}")
    print(f"  Macro Accuracy:  {val_macro_acc:.4f}")
    print(f"  Macro F1:        {val_f1_macro:.4f}")
    print(f"  Weighted F1:     {val_f1_weighted:.4f}")

    print("\nTest metrics:")
    print(f"  Macro Precision: {test_precision:.4f}")
    print(f"  Macro Recall:    {test_recall:.4f}")
    print(f"  Macro Accuracy:  {test_macro_acc:.4f}")
    print(f"  Macro F1:        {test_f1_macro:.4f}")
    print(f"  Weighted F1:     {test_f1_weighted:.4f}")

    mlflow.set_tracking_uri(cfg.baseline.logging.mlflow_uri)
    mlflow.set_experiment(cfg.baseline.logging.experiment_name)
    with mlflow.start_run(run_name=cfg.baseline.logging.run_name):
        mlflow.log_params(
            {
                "model_type": "baseline_tfidf_logreg",
                **{f"tfidf_{k}": v for k, v in cfg.baseline.tfidf.items()},
                **{f"logreg_{k}": v for k, v in cfg.baseline.logreg.items()},
            }
        )
        mlflow.log_metrics(
            {
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_macro_accuracy": val_macro_acc,
                "val_f1_macro": val_f1_macro,
                "val_f1_weighted": val_f1_weighted,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_macro_accuracy": test_macro_acc,
                "test_f1_macro": test_f1_macro,
                "test_f1_weighted": test_f1_weighted,
            }
        )
        save_dir = Path(cfg.baseline.model.save_dir)
        model.save(save_dir)
        mlflow.log_artifacts(str(save_dir), artifact_path="baseline_model")


if __name__ == "__main__":
    train()
