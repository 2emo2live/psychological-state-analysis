import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from psychology_state_analyzer.data_processing.dataset import MentalHealthDataset


class MentalHealthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        tokenizer_name: str,
        batch_size: int,
        max_len: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _collate_fn(self, batch):
        texts, labels = zip(*batch)
        encoding = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = MentalHealthDataset(self.hparams.root_path, "train")
            self.val_dataset = MentalHealthDataset(self.hparams.root_path, "val")
        if stage in ("test", None):
            self.test_dataset = MentalHealthDataset(self.hparams.root_path, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
