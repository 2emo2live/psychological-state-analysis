import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from transformers import DistilBertTokenizer

from psychology_state_analyzer.data_processing.datamodule import MentalHealthDataModule
from psychology_state_analyzer.data_processing.load_data import load_data
from psychology_state_analyzer.models.main_model import PsychologicalStateModel


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train the segmentation model using config from Hydra.

    Args:
        cfg: Hydra config composed from conf/config.yaml and its defaults.
    """
    pl.seed_everything(cfg.train.seed, workers=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(cfg.data.dvc_path)

    tokenizer = DistilBertTokenizer.from_pretrained(cfg.model.backbone)

    max_len = tokenizer.model_max_length  # TODO: explore

    datamodule = MentalHealthDataModule(
        train_texts=X_train,
        train_labels=y_train,
        val_texts=X_val,
        val_labels=y_val,
        test_texts=X_test,
        test_labels=y_test,
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        max_len=max_len,
    )

    model = PsychologicalStateModel(
        model_name=cfg.model.backbone,
        num_classes=7,
        num_layers_to_train=cfg.model.num_layers_to_train,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
    )

    """callbacks = [
        ModelCheckpoint(
            dirpath=cfg.train.checkpoint_dir,
            filename="best",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=cfg.train.checkpoint_dir,
            filename="epochs",
            save_top_k=cfg.train.epochs,
            every_n_epochs=1,
        )
    ]"""

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=True,
        logger=mlf_logger,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        # callbacks=callbacks,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        # ckpt_path=cfg.train.resume,
    )


if __name__ == "__main__":
    train()
