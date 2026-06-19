import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import MentalHealthDataset


class MentalHealthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
        tokenizer,
        batch_size: int = 32,
        max_len: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Создаем датасеты для каждого этапа
        self.train_dataset = MentalHealthDataset(
            self.hparams.train_texts,
            self.hparams.train_labels,
            self.hparams.tokenizer,
            self.hparams.max_len,
        )
        self.val_dataset = MentalHealthDataset(
            self.hparams.val_texts,
            self.hparams.val_labels,
            self.hparams.tokenizer,
            self.hparams.max_len,
        )
        self.test_dataset = MentalHealthDataset(
            self.hparams.test_texts,
            self.hparams.test_labels,
            self.hparams.tokenizer,
            self.hparams.max_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)
