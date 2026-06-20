from pathlib import Path
from typing import List, Optional

import fire
import hydra
from omegaconf import DictConfig

from psychology_state_analyzer.train_models.train_baseline import (
    train as train_baseline,
)
from psychology_state_analyzer.train_models.train_main_model import train as train_main
from psychology_state_analyzer.utils.export_to_onnx import export_to_onnx

# from psychology_state_analyzer.utils.export_to_tensorrt import export_to_tensorrt


class Commands:
    def __init__(self):
        # Определяем путь к папке configs относительно этого файла
        self.config_dir = "../configs"

    def _compose_config(
        self, config_name: str = "config", overrides: Optional[List[str]] = None
    ) -> DictConfig:
        """Загружает конфиг через Hydra без декоратора."""
        with hydra.initialize(version_base=None, config_path=self.config_dir):
            cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
        return cfg

    def train_main(self, overrides: Optional[List[str]] = None):
        """
        Запуск обучения основной модели (DistilBERT).
        """
        cfg = self._compose_config(overrides=overrides)
        train_main(cfg)

    def train_baseline(self, overrides: Optional[List[str]] = None):
        """
        Запуск обучения бейзлайна (TF‑IDF + логистическая регрессия).
        """
        cfg = self._compose_config(overrides=overrides)
        train_baseline(cfg)

    def export_onnx(
        self,
        overrides: Optional[List[str]] = None,
    ):
        """
        Экспорт лучшей модели в ONNX.
        """
        cfg = self._compose_config(overrides=overrides)
        export_to_onnx(cfg)

    def export_tensorrt(
        self,
        overrides: Optional[List[str]] = None,
    ):
        """
        Конвертация ONNX в TensorRT engine.
        """
        cfg = self._compose_config(overrides=overrides)
        # export_to_tensorrt(cfg)
        pass


if __name__ == "__main__":
    fire.Fire(Commands)
