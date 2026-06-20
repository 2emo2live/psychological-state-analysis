import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score, Precision, Recall
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup,
)


class PsychologicalStateModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_layers_to_train: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, num_classes)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_layers = self.backbone.config.num_hidden_layers
        assert num_layers_to_train <= num_layers, (
            "num_layers_to_train must be <= num_layers"
        )
        for i in range(num_layers - num_layers_to_train, num_layers):
            for name, param in self.backbone.named_parameters():
                if f"layer.{i}" in name:
                    param.requires_grad = True

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.val_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_weighted_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )

        self.test_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_weighted_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )

    def forward(self, input_ids, attention_mask):
        embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = embeddings.last_hidden_state[:, 0, :]
        logits = self.head(cls_embedding)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(outputs, batch["labels"])
        preds = torch.argmax(outputs, dim=1)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc(preds, batch["labels"])
        self.log(
            "train_accuracy", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(outputs, batch["labels"])
        preds = torch.argmax(outputs, dim=1)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_accuracy(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])
        self.val_recall(preds, batch["labels"])
        self.val_precision(preds, batch["labels"])
        self.val_weighted_f1(preds, batch["labels"])
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_epoch=True, prog_bar=True)
        self.log("val_weighted_f1", self.val_weighted_f1, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.loss_fn(outputs, batch["labels"])
        preds = torch.argmax(outputs, dim=1)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.test_accuracy(preds, batch["labels"])
        self.test_f1(preds, batch["labels"])
        self.test_recall(preds, batch["labels"])
        self.test_precision(preds, batch["labels"])
        self.test_weighted_f1(preds, batch["labels"])
        self.log("test_accuracy", self.test_accuracy, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)
        self.log("test_recall", self.test_recall, on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision, on_epoch=True, prog_bar=True)
        self.log(
            "test_weighted_f1", self.test_weighted_f1, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
