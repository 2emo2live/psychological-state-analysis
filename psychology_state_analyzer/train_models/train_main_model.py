from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from psychology_state_analyzer.data_processing.datamodule import MentalHealthDataModule
from psychology_state_analyzer.data_processing.load_data import load_data
from psychology_state_analyzer.models.main_model import PsychologicalStateModel


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed, workers=True)

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

    datamodule = MentalHealthDataModule(
        root_path=cfg.data.root_path,
        tokenizer_name=cfg.model.backbone,
        batch_size=cfg.data.batch_size,
        max_len=cfg.data.max_len,
    )

    model = PsychologicalStateModel(
        model_name=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        num_layers_to_train=cfg.model.num_layers_to_train,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.train.checkpoint_dir,
            filename="best",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=cfg.train.checkpoint_dir,
            filename="last",
            save_top_k=1,
            every_n_epochs=1,
        ),
    ]

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=True,
        logger=mlf_logger,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        callbacks=callbacks,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        # ckpt_path=cfg.train.resume,
    )

    best_model_path = Path(cfg.train.checkpoint_dir) / "best.ckpt"
    test_results = trainer.test(
        datamodule=datamodule, ckpt_path=best_model_path, verbose=True
    )


if __name__ == "__main__":
    train()
