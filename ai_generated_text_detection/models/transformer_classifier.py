import math

import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MetricCollection
import pytorch_lightning as pl

from ai_generated_text_detection.logger import logger


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, : x.size(1), :]


class SimpleTransformerClassifier(nn.Module):
    def __init__(
            self,
            vocab_size,
            max_len=200,
            embedding_dim=128,
            num_heads=4,
            num_layers=2,
            hidden_dim=256,
            dropout=0.1,
            num_classes=2  # ДОБАВЛЕНО
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes

        # 1. Эмбеддинг слов
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. Позиционное кодирование
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Классификатор - выход зависит от num_classes
        if num_classes == 2:
            output_dim = 1  # Для бинарной классификации один выход
        else:
            output_dim = num_classes  # Для многоклассовой

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

        # 5. Инициализация
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 1. Эмбеддинг слов
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]

        # 2. Добавляем позиционное кодирование
        x = self.pos_encoder(x)

        # 3. Создаем маску для padding токенов
        # padding_mask: True для padding токенов
        padding_mask = (x.sum(dim=-1) == 0)  # все нули в эмбеддинге

        # 4. Пропускаем через Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 5. Берем среднее по всем токенам (кроме padding)
        mask = ~padding_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_x = x * mask.float()
        mean_pooled = masked_x.sum(dim=1) / mask.sum(dim=1).float()

        # 6. Классификация
        logits = self.classifier(mean_pooled)

        return logits


class TransformerEssayClassifier(pl.LightningModule):
    def __init__(self, vocab_size, learning_rate=1e-4, **model_params):
        super().__init__()

        # Сохраняем все гиперпараметры
        self.save_hyperparameters(ignore=['model_params'])

        # Извлекаем параметры модели из model_params
        embedding_dim = model_params.get('embedding_dim', 128)
        num_heads = model_params.get('num_heads', 4)
        num_layers = model_params.get('num_layers', 3)
        hidden_dim = model_params.get('hidden_dim', 256)
        dropout = model_params.get('dropout', 0.1)
        num_classes = model_params.get('num_classes', 2)
        max_len = model_params.get('max_len', 200)

        # Модель с параметрами из конфига
        self.model = SimpleTransformerClassifier(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes
        )

        self.lr = learning_rate
        self.num_classes = num_classes

        # Настраиваем метрики в зависимости от задачи
        task = "binary" if num_classes == 2 else "multiclass"
        num_labels = num_classes if num_classes > 2 else None

        self.train_metrics = MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task=task, num_classes=num_labels),
                "precision": torchmetrics.Precision(task=task, num_classes=num_labels),
                "recall": torchmetrics.Recall(task=task, num_classes=num_labels),
                "f1": torchmetrics.F1Score(task=task, num_classes=num_labels),
                "auc": torchmetrics.AUROC(task=task, num_classes=num_labels) if task == "binary" else None,
            },
            prefix="train_",
        )

        # Убираем None значения
        self.train_metrics = MetricCollection(
            {k: v for k, v in self.train_metrics.items() if v is not None}
        )

        self.val_metrics = MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task=task, num_classes=num_labels),
                "precision": torchmetrics.Precision(task=task, num_classes=num_labels),
                "recall": torchmetrics.Recall(task=task, num_classes=num_labels),
                "f1": torchmetrics.F1Score(task=task, num_classes=num_labels),
                "auc": torchmetrics.AUROC(task=task, num_classes=num_labels) if task == "binary" else None,
            },
            prefix="val_",
        )

        # Убираем None значения
        self.val_metrics = MetricCollection(
            {k: v for k, v in self.val_metrics.items() if v is not None}
        )

    def forward(self, x):
        return self.model(x)

    def _calculate_loss_and_preds(self, y_hat, y):
        if self.num_classes == 2:
            # Для бинарной классификации
            if y_hat.shape[1] == 1:
                loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.float())
                probs = torch.sigmoid(y_hat).squeeze()
                preds = (probs > 0.5).float()
            else:
                loss = F.cross_entropy(y_hat, y.long())
                probs = torch.softmax(y_hat, dim=1)
                preds = torch.argmax(y_hat, dim=1).float()
        else:
            # Для многоклассовой классификации
            loss = F.cross_entropy(y_hat, y.long())
            probs = torch.softmax(y_hat, dim=1)
            preds = torch.argmax(y_hat, dim=1).float()

        return loss, preds, probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, preds, probs = self._calculate_loss_and_preds(y_hat, y)

        # Обновляем все метрики сразу
        if self.num_classes == 2:
            metrics = self.train_metrics(preds, y, probs=probs)
        else:
            metrics = self.train_metrics(preds, y)

        self.log_dict(metrics, prog_bar=False)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, preds, probs = self._calculate_loss_and_preds(y_hat, y)

        # Обновляем все метрики
        if self.num_classes == 2:
            metrics = self.val_metrics(preds, y, probs=probs)
        else:
            metrics = self.val_metrics(preds, y)

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