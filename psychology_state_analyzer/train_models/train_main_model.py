# import hydra
import pytorch_lightning as pl

# from loggers.resolver import get_logger
# from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import DistilBertTokenizer

from psychology_state_analyzer.data_processing.datamodule import MentalHealthDataModule
from psychology_state_analyzer.data_processing.load_data import load_data
from psychology_state_analyzer.models.main_model import PsychologicalStateModel


# @hydra.main(config_path="../../conf", config_name="config", version_base=None)
# def train(cfg: DictConfig) -> None:
def train() -> None:
    """Train the segmentation model using config from Hydra.

    Args:
        cfg: Hydra config composed from conf/config.yaml and its defaults.
    """
    # pl.seed_everything(cfg.training.seed, workers=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data("data/Data.csv")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    datamodule = MentalHealthDataModule(
        train_texts=X_train,
        train_labels=y_train,
        val_texts=X_val,
        val_labels=y_val,
        test_texts=X_test,
        test_labels=y_test,
        tokenizer=tokenizer,
        batch_size=32,
        max_len=128,
    )

    model = PsychologicalStateModel(
        model_name="distilbert-base-uncased",
        num_classes=7,
        num_layers_to_train=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        warmup_steps=100,
        device="cuda",
    )

    """callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="best",
            monitor="val_accuracy",
            mode="max",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="last",
            save_top_k=1,
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]"""

    trainer = pl.Trainer(
        max_epochs=1,
        # logger=get_logger(cfg),
        log_every_n_steps=100,
        deterministic=False,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        # ckpt_path=cfg.training.resume,
    )


if __name__ == "__main__":
    print(1)
    train()
