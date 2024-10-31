import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(32 * 5 * 5, 120),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = self.linear1(x2)
        x4 = self.linear2(x3)
        output = self.linear3(x4)

        return output
