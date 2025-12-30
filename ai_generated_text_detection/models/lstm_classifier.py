import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=128,
        num_layers=1,
        dropout=0.5,
        num_classes=2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Изменяем выходной слой для поддержки num_classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes if num_classes > 2 else 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)  # [batch_size, num_classes] или [batch_size, 1]
        return output


class EssayLSTMClassifier(pl.LightningModule):
    def __init__(self, vocab_size, learning_rate=1e-3, **model_params):
        super().__init__()

        # Сохраняем все гиперпараметры
        self.save_hyperparameters(ignore=["model_params"])

        # Извлекаем параметры модели из model_params
        embedding_dim = model_params.get("embedding_dim", 256)
        hidden_dim = model_params.get("hidden_dim", 128)
        num_layers = model_params.get("num_layers", 1)
        dropout = model_params.get("dropout", 0.5)
        num_classes = model_params.get("num_classes", 2)

        # Создаем модель с параметрами из конфига
        self.model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.lr = learning_rate
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Для бинарной классификации (num_classes=2)
        if self.num_classes <= 2:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1))
        else:
            loss = F.cross_entropy(y_hat, y.long())

        self.log("train_loss", loss, prog_bar=True)

        if (
            self.trainer.current_epoch % 10 == 0
            or self.trainer.current_epoch == self.trainer.max_epochs - 1
        ):
            self.log(
                f"Epoch {self.trainer.current_epoch}, Batch {batch_idx}: Train loss = {loss:.4f}"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Для бинарной классификации
        if self.num_classes <= 2:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1))
            preds = (torch.sigmoid(y_hat) > 0.5).float()
        else:
            loss = F.cross_entropy(y_hat, y.long())
            preds = torch.argmax(y_hat, dim=1).float()

        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        if (
            self.trainer.current_epoch % 10 == 0
            or self.trainer.current_epoch == self.trainer.max_epochs - 1
        ):
            self.log(
                f"Epoch {self.trainer.current_epoch}, Batch {batch_idx}: Val acc = {acc}"
            )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
