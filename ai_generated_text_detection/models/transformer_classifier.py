import math

import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MetricCollection
import pytorch_lightning as pl

from ai_generated_text_detection.logger import logger


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, : x.size(1), :]


class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=200,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # 1. Эмбеддинг слов
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 2. Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # важно!
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 5. Инициализация
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 1. Эмбеддинг слов
        x = self.embedding(x)  # [batch, seq_len, d_model]

        # 2. Добавляем позиционное кодирование
        x = self.pos_encoder(x)

        # 3. Создаем маску для padding токенов
        # padding_mask: True для padding токенов
        padding_mask = x.sum(dim=-1) == 0  # все нули в эмбеддинге

        # 4. Пропускаем через Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 5. Берем первый токен (как в BERT) или среднее
        # Вариант 1: Первый токен (CLS)
        # cls_token = x[:, 0, :]

        # Вариант 2: Среднее по всем токенам (кроме padding)
        mask = ~padding_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_x = x * mask.float()
        mean_pooled = masked_x.sum(dim=1) / mask.sum(dim=1).float()

        # 6. Классификация
        logits = self.classifier(mean_pooled)

        return logits


class TransformerEssayClassifier(pl.LightningModule):
    def __init__(self, vocab_size, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Модель
        self.model = SimpleTransformerClassifier(
            vocab_size=vocab_size,
            max_len=200,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.1,
        )

        self.lr = learning_rate

        # Все метрики в одном объекте
        self.train_metrics = MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "auc": torchmetrics.AUROC(task="binary"),
            },
            prefix="train_",
        )

        self.val_metrics = MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "auc": torchmetrics.AUROC(task="binary"),
            },
            prefix="val_",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Обновляем все метрики сразу
        preds = (torch.sigmoid(y_hat) > 0.5).float()
        probs = torch.sigmoid(y_hat)

        metrics = self.train_metrics(preds, y, probs=probs)
        self.log_dict(metrics, prog_bar=False)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # Обновляем все метрики
        preds = (torch.sigmoid(y_hat) > 0.5).float()
        probs = torch.sigmoid(y_hat)

        metrics = self.val_metrics(preds, y, probs=probs)
        self.log_dict(metrics, prog_bar=False)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # Печатаем все метрики каждые 5 эпох
        if (self.current_epoch + 1) % 5 == 0:
            metrics = self.val_metrics.compute()
            logger.info(f"\nЭпоха {self.current_epoch + 1} - Validation:")
            for name, value in metrics.items():
                logger.info(f"  {name.replace('val_', '').title()}: {value:.4f}")

        # Сбрасываем метрики
        self.train_metrics.reset()
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
