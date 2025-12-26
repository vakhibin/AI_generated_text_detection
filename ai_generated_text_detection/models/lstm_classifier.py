import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=128, num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 потому что bidirectional

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)  # [batch_size, 1]
        return output


class EssayLSTMClassifier(pl.LightningModule):
    def __init__(self, vocab_size, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = LSTMClassifier(
            vocab_size=vocab_size, embedding_dim=256, hidden_dim=128, num_layers=1
        )
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

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
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        preds = (torch.sigmoid(y_hat) > 0.5).float()
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
