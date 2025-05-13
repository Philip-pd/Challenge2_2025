import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMLP(nn.Module):
    def __init__(self, n_steps, n_mels, output_size, time_reduce=1,
                 hidden1_size=512, hidden2_size=256, hidden3_size=128):
        super().__init__()

        # 2D CNN feature extractor (Time x Mels → Feature Maps)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (n_steps/2, n_mels/2)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (n_steps/4, n_mels/4)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (n_steps/8, n_mels/8)
        )

        # Compute CNN output shape
        dummy_input = torch.zeros(1, 1, n_steps, n_mels)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
            flattened_size = cnn_out.view(1, -1).shape[1]

        # MLP head
        self.fc1 = nn.Linear(flattened_size, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.bn3 = nn.BatchNorm1d(hidden3_size)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hidden3_size, 64)  # Extra hidden layer
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)

        self.output = nn.Linear(64, output_size)

    def forward(self, x):
        # x: (batch_size, n_steps, n_mels) → (batch_size, 1, n_steps, n_mels)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = self.output(x)
        return x