import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score
from transformers import (
    AutoModelForSequenceClassification,
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

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

        for name, param in self.model.distilbert.named_parameters():
            param.requires_grad = False

        num_layers = self.model.config.num_hidden_layers
        assert num_layers_to_train <= num_layers, (
            "num_layers_to_train must be less than or equal to num_layers"
        )
        for i in range(num_layers - num_layers_to_train, num_layers):
            layer_name = f"transformer.layer.{i}"
            for name, param in self.model.distilbert.named_parameters():
                if layer_name in name:
                    param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_accuracy(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

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
