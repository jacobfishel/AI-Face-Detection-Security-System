
import torch
import torch.nn as nn

class WiderFaceNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.R = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 10, 5, 1, 2)
        self.conv2 = nn.Conv2d(10, 20, 5, 1, 2)
        self.conv3 = nn.Conv2d(20, 30, 5, 1, 2)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten_dim = None
        self.fc1 = None
        # self.fc1 = nn.Linear(122880, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 500)  # 100 boxes x (4 + 1 confidence)

    def _get_flatten_size(self, x):
        x = self.pool(self.R(self.conv1(x)))
        x = self.pool(self.R(self.conv2(x)))
        x = self.pool(self.R(self.conv3(x)))
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        if self.flatten_dim is None:
            self.flatten_dim = self._get_flatten_size(x)
            self.fc1 = nn.Linear(self.flatten_dim, 1024).to(x.device)

        x = self.pool(self.R(self.conv1(x)))
        x = self.pool(self.R(self.conv2(x)))
        x = self.pool(self.R(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.R(self.fc1(x))
        x = self.R(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), 100, 5)  # [batch, 100 boxes, 5 values]

        pred_boxes = x[:, :, :4]
        conf_scores = x[:, :, 4]

        return pred_boxes, conf_scores
