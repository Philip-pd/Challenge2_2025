import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, p=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AudioMLP(nn.Module):
    def __init__(self, n_steps, n_mels, output_size, time_reduce=1,
                 hidden1_size=256, hidden2_size=128, hidden3_size=64, num_classes=50, block_drop_p=0.2, head_drop_p=0.5):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout0 = nn.Dropout2d(block_drop_p)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, p=block_drop_p)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, p=block_drop_p)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, p=block_drop_p)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, p=block_drop_p)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head_dropout = nn.Dropout(head_drop_p)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, p):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, p))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dropout0(self.relu(self.bn1(self.conv1(x))))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.head_dropout(out)
        return self.fc(out)

