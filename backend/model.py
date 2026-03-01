# model.py
import torch
import torch.nn as nn

class ASLSignLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths=None):
        # x: (batch, max_seq_len, 63)
        if lengths is not None:
            # Pack so the LSTM ignores padding frames
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        lstm_out, (hidden, _) = self.lstm(x)
        # hidden[-1] is the last layer's final hidden state: (batch, hidden_size)
        return self.classifier(hidden[-1])