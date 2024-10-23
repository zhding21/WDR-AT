import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=2, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3,  stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.Conv5(x4)
        # x5 = x5.view(x5.size(0), 256 * 3 * 3)
        x6 = self.linear1(x5)
        x7 = self.linear2(x6)
        output = self.linear3(x7)

        return output
