"""Универсальный Lightning модуль для обучения и инференса моделей."""

import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
import torchmetrics
from typing import Optional, Dict, Any

from ai_generated_text_detection.logger import logger


class UniversalModelModule(L.LightningModule):
    """
    Универсальный Lightning модуль для обучения и инференса любых PyTorch моделей.

    Parameters
    ----------
    model : nn.Module
        PyTorch модель для обучения.
    model_type : str
        Тип модели ("lstm", "transformer", "baseline").
    learning_rate : float
        Learning rate для оптимизатора.
    weight_decay : float, optional
        Weight decay для оптимизатора (по умолчанию 0.0).
    optimizer : str, optional
        Тип оптимизатора ("adam", "adamw", "sgd") (по умолчанию "adam").
    scheduler : str, optional
        Тип scheduler ("reduce_lr_on_plateau", "cosine", "step", None) (по умолчанию None).
    scheduler_patience : int, optional
        Patience для ReduceLROnPlateau (по умолчанию 5).
    scheduler_factor : float, optional
        Factor для ReduceLROnPlateau (по умолчанию 0.1).
    log_interval : int, optional
        Интервал логирования детальных метрик (по умолчанию 10).
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "custom",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.1,
        log_interval: int = 10,
    ):
        super().__init__()

        self.model = model
        self.model_type = model_type.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer.lower()
        self.scheduler_type = scheduler.lower() if scheduler else None
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.log_interval = log_interval

        # Сохраняем гиперпараметры для воспроизводимости
        self.save_hyperparameters(ignore=["model"])

        # Функция потерь для бинарной классификации
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Инициализация метрик
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Инициализирует метрики для трейна и валидации."""

        # Общие метрики для бинарной классификации
        common_metrics = {
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
            "auc": torchmetrics.AUROC(task="binary"),
        }

        # Метрики для тренировки
        self.train_metrics = MetricCollection({**common_metrics}, prefix="train_")

        # Метрики для валидации
        self.val_metrics = MetricCollection({**common_metrics}, prefix="val_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель.

        Parameters
        ----------
        x : torch.Tensor
            Входные данные.

        Returns
        -------
        torch.Tensor
            Выход модели.
        """
        return self.model(x)

    def _common_step(self, batch: tuple, stage: str) -> torch.Tensor:
        """
        Общий шаг для тренировки и валидации.

        Parameters
        ----------
        batch : tuple
            Батч данных (x, y).
        stage : str
            Стадия ("train", "val").

        Returns
        -------
        torch.Tensor
            Loss.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        # Вычисляем предсказания и вероятности
        probs = torch.sigmoid(y_hat)
        preds = (probs > 0.5).float()

        # Обновляем метрики
        if stage == "train":
            self.train_metrics(preds, y, probs=probs)
        else:  # val
            self.val_metrics(preds, y, probs=probs)

        # Логируем loss
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Шаг тренировки.

        Parameters
        ----------
        batch : tuple
            Батч данных (x, y).
        batch_idx : int
            Индекс батча.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = self._common_step(batch, "train")

        # Дополнительное логирование для LSTM
        if self.model_type == "lstm" and (
            self.trainer.current_epoch % self.log_interval == 0
            or self.trainer.current_epoch == self.trainer.max_epochs - 1
        ):
            self.log(
                f"Epoch {self.trainer.current_epoch}, Batch {batch_idx}: Train loss = {loss:.4f}",
                logger=True,
            )

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Шаг валидации.

        Parameters
        ----------
        batch : tuple
            Батч данных (x, y).
        batch_idx : int
            Индекс батча.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        loss = self._common_step(batch, "val")

        # Дополнительное логирование для LSTM
        if self.model_type == "lstm" and (
            self.trainer.current_epoch % self.log_interval == 0
            or self.trainer.current_epoch == self.trainer.max_epochs - 1
        ):
            preds = (torch.sigmoid(self(batch[0])) > 0.5).float()
            acc = (preds == batch[1]).float().mean()
            self.log(
                f"Epoch {self.trainer.current_epoch}, Batch {batch_idx}: Val acc = {acc:.4f}",
                logger=True,
            )

        return loss

    def on_train_epoch_end(self) -> None:
        """
        Вызывается в конце эпохи тренировки.
        Логирует метрики и сбрасывает их.
        """
        if self.current_epoch % self.log_interval == 0:
            metrics = self.train_metrics.compute()
            self._log_metrics(metrics, "train")

        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """
        Вызывается в конце эпохи валидации.
        Логирует метрики и сбрасывает их.
        """
        if self.current_epoch % self.log_interval == 0:
            metrics = self.val_metrics.compute()
            self._log_metrics(metrics, "val")

        self.val_metrics.reset()

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], stage: str) -> None:
        """
        Логирует метрики.

        Parameters
        ----------
        metrics : Dict[str, torch.Tensor]
            Словарь с метриками.
        stage : str
            Стадия ("train", "val").
        """
        logger.info(f"\n📊 Эпоха {self.current_epoch + 1} - {stage.title()}:")
        for name, value in metrics.items():
            clean_name = name.replace(f"{stage}_", "").title()
            logger.info(f"  {clean_name}: {value:.4f}")
            self.log(name, value, logger=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Конфигурирует оптимизатор и scheduler.

        Returns
        -------
        Dict[str, Any]
            Конфигурация оптимизатора и scheduler.
        """
        # Выбор оптимизатора
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Неизвестный оптимизатор: {self.optimizer_type}")

        # Конфигурация scheduler
        if self.scheduler_type == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.scheduler_patience,
                factor=self.scheduler_factor,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
            return [optimizer], [scheduler]
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.trainer.max_epochs // 3, gamma=0.1
            )
            return [optimizer], [scheduler]
        else:
            # Без scheduler
            return optimizer

    def predict_step(
        self, batch: tuple, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Шаг предсказания для инференса.

        Parameters
        ----------
        batch : tuple
            Батч данных (x, y).
        batch_idx : int
            Индекс батча.
        dataloader_idx : int
            Индекс даталоадера.

        Returns
        -------
        Dict[str, torch.Tensor]
            Словарь с предсказаниями, вероятностями и истинными метками.
        """
        x, y = batch
        with torch.no_grad():
            y_hat = self.model(x)
            probs = torch.sigmoid(y_hat)
            preds = (probs > 0.5).float()

        return {"predictions": preds, "probabilities": probs, "labels": y}
