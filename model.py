from torch import nn
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import AutoModelForTokenClassification
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from datetime import datetime
import pandas as pd


class DiceLoss(nn.Module):
    def __init__(self, class_weights=[0.003, 1, 2, 2]):
        super(DiceLoss, self).__init__()
        self.register_buffer("class_weights", torch.Tensor(class_weights))

    def forward(self, input, target):
        probs = F.softmax(input, 1)
        target = F.one_hot(target, 4)
        tp = (probs * target).sum(0)
        per_class = (2 * tp + 0.001) / (target.sum(0) + probs.sum(0) + 0.001)
        return self.class_weights.sum() - (per_class * self.class_weights).sum()


def average_per_word_logits(token_logits: torch.Tensor, words_mapping: torch.Tensor) -> torch.Tensor:
    """
    Aggregates per-token logits and returns average per-word logits.

    Args:
        token_logits (torch.Tensor): A tensor of shape (batch, token, class).
        words_mapping (torch.Tensor): A tensor of shape (batch, token) with indices indicating to which word a token belongs.

    Returns:
        torch.Tensor: A tensor of shape (batch, word) with average logits per word.
    """
    batch_size, num_tokens, num_classes = token_logits.shape
    max_word_index = words_mapping.max().item()

    # Create a mask for valid tokens (ignore tokens mapped to -1)
    valid_mask = words_mapping != -1

    # Create a one-hot encoding of the words_mapping tensor
    word_one_hot = torch.nn.functional.one_hot(words_mapping * valid_mask, num_classes=max_word_index + 1).float()

    # Use the valid_mask to zero out the invalid tokens in word_one_hot
    word_one_hot = word_one_hot * valid_mask.unsqueeze(-1).float()

    # Sum the logits per word
    word_logits_sum = torch.matmul(word_one_hot.transpose(1, 2), token_logits)

    # Count the tokens per word
    word_counts = word_one_hot.sum(dim=1).unsqueeze(-1)

    # Avoid division by zero
    word_counts[word_counts == 0] = 1

    # Compute the average logits per word
    average_word_logits = word_logits_sum / word_counts

    return average_word_logits


class MyModel(pl.LightningModule):
    # Класс с моделью - описывает forward pass, а также что делать на
    # разных стадиях обучения - логгирование, расчет метрик, сохранение предсказаний
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_loss = 0
        self.train_loss_cnt = 0
        self.model = AutoModelForTokenClassification.from_pretrained(
            cfg.model.name, num_labels=4, ignore_mismatched_sizes=True
        )
        # self.loss = nn.CrossEntropyLoss()
        self.loss = DiceLoss()

    def forward(self, batch):
        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        return average_per_word_logits(logits, batch["words_mapping"])

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        logits = logits.reshape(-1, 4)
        labels = batch["word_labels"].flatten()
        logits = logits[labels != -100]
        labels = labels[labels != -100]
        loss = self.loss(logits, labels)
        self.train_loss += loss.item()
        self.train_loss_cnt += 1
        if self.train_loss_cnt == self.trainer.log_every_n_steps * self.trainer.accumulate_grad_batches:
            self.log("train/loss", self.train_loss / self.train_loss_cnt, on_step=True)
            self.train_loss = 0
            self.train_loss_cnt = 0
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        logits = logits.reshape(-1, 4)
        labels = batch["word_labels"].flatten()
        text_ids = np.repeat(batch["text_ids"], batch["word_labels"].shape[1])
        logits = logits[labels != -100]
        text_ids = text_ids[labels.cpu().numpy() != -100]
        labels = labels[labels != -100]
        loss = self.loss(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.preds.append(logits.cpu().numpy())
        self.labels.append(labels.cpu().numpy())
        self.text_ids.append(text_ids)

    def on_validation_epoch_start(self) -> None:
        self.preds = []
        self.labels = []
        self.text_ids = []

    def on_validation_epoch_end(self) -> None:
        # if len(self.preds[0]) < 10:
        #     return
        text_ids = np.concatenate(self.text_ids)
        preds = np.concatenate(self.preds)
        labels = np.concatenate(self.labels)
        f1s = f1_score(labels, preds.argmax(1), average=None, labels=[0, 1, 2, 3])
        self.log("val/f1_o", f1s[0], prog_bar=False)
        self.log("val/f1_bdiscount", f1s[1], prog_bar=False)
        self.log("val/f1_bvalue", f1s[2], prog_bar=False)
        self.log("val/f1_ivalue", f1s[3], prog_bar=False)
        self.log("val/f1", f1s[0] * 0.003 + f1s[1] * 1 + f1s[2] * 2 + f1s[3] * 2, prog_bar=True)
        # if self.current_epoch == self.trainer.max_epochs - 1:
        #     ckpt_name = (
        #         self.cfg.model.name.lower().replace("/", "-")
        #         + datetime.now().strftime("_%Y%m%d_%H%M%S")
        #         + "fold%d" % self.cfg.data.fold
        #     )
        #     pd.DataFrame(
        #         {"text_id": text_ids, "p0": preds[:, 0], "p1": preds[:, 1], "p2": preds[:, 2], "p3": preds[:, 3]}
        #     ).to_parquet(f"preds/{ckpt_name}.pq")

    def configure_optimizers(self):
        def no_decay(n):
            return any(nd in n for nd in ["bias", "LayerNorm.weight"])

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not no_decay(n)],
                "weight_decay": 0.01,
                "lr": self.cfg.model.learning_rate,
            },
            {
                "params": [p for n, p in self.named_parameters() if no_decay(n)],
                "weight_decay": 0.0,
                "lr": self.cfg.model.learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-6)
        return optimizer
