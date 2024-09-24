import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import json
import os


class cnn_model_1_conv(nn.Module):  # REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_1_conv, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            # 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),

            # 3x3, second part
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.conv1x1 = nn.Conv2d(256, 1, kernel_size=1, bias=False)  # Reduce to 1 channel

        # Output size calculation based on input dimensions
        self.fc_input_size = 1 * 16 * 18
        self.fc = nn.Linear(self.fc_input_size, 3 * output_w)

        self.output_w = output_w

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1x1(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 1, 3, self.output_w)  # Reshape to (batch_size, 1, 3, output_w)
        return x


class ImprovedFixedResNetSParamPredictor(nn.Module):
    def __init__(self, output_w):
        super(ImprovedFixedResNetSParamPredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(0.05)

        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2b = nn.BatchNorm2d(64)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3b = nn.BatchNorm2d(128)

        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4b = nn.BatchNorm2d(256)

        self.conv5a = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5a = nn.BatchNorm2d(256)
        self.conv5b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5b = nn.BatchNorm2d(256)

        self.conv1x1 = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.fc_input_size = 1 * 16 * 18
        self.fc = nn.Linear(self.fc_input_size, 3 * output_w)

        self.output_w = output_w

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        residual = x
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x += residual

        residual = x
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x += residual

        residual = x
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.bn4b(self.conv4b(x))
        x += residual

        residual = x
        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.bn5b(self.conv5b(x))
        x += residual

        x = self.conv1x1(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 1, 3, self.output_w)  # Reshape to (batch_size, 1, 3, output_w)
        return x
class SParamPredictor(nn.Module):
    def __init__(self, output_w):
        super(SParamPredictor, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.05),

            # 3x3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            # 3x3, second part
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.3)
        )
        self.conv1x1 = nn.Conv2d(128, 1, kernel_size=1, bias=False)  # Reduce to 1 channel

        # Output size calculation based on input dimensions
        self.fc_input_size = 1 * 16 * 18
        self.fc = nn.Linear(self.fc_input_size, 3 * output_w)

        self.output_w = output_w

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1x1(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 1, 3, self.output_w)  # Reshape to (batch_size, 1, 3, output_w)
        return x
class cnn_model_elu(nn.Module):  #REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_elu, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            # 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),

            # 3x3, second part
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 18, 1000),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(500, 3 * output_w),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x
class cnn_model(nn.Module):  #REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            # 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),

            # 3x3, second part
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 18, 1000),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(500, 3 * output_w),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class cnn_model_short(nn.Module):  # REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_short, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            # 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),

            # 3x3, second part
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 18, 1000),  # input size is 16x18 after conv layers
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(500, 3 * output_w),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

import torch
import torch.nn as nn

class skip_and_pool(nn.Module):  # REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(skip_and_pool, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1000),  # Adjust the input size based on the pooling layers
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(500, 3 * output_w)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x  # Skip connection
        x = self.conv3(x)
        x = self.conv4(x) + x  # Skip connection
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class cnn_model_more_conv(nn.Module):  #REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_more_conv, self).__init__()
        self.conv = nn.Sequential(
            #7x7
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 5x5
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            # 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),

            # 3x3, second part
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),

        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 18, 1000),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(200, 3 * output_w),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class cnn_model_1(nn.Module):  # REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_1, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            # 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            # 3x3, second part
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 10, 1000),  # input size is 16x18 after conv layers
            nn.BatchNorm1d(1000),
            nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(500, 3 * output_w),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x




class cnn_model_og_study(nn.Module):  #REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_og_study, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 64, kernel_size=12, stride=1, padding=6), #17x19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=10, stride=1, padding=5),  # 18x20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=4),  # 19x21
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=6, stride=1, padding=2),  # 18x20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 18x20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 18x20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=1),  # 17x19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=1),  # 16x18
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),  # 17x19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 17x19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 17x19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 17x19
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 17 * 19, 500),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            nn.Linear(500, 500),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            nn.Linear(500, 500),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            nn.Linear(500, 500),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),

            nn.Linear(500, 3 * output_w),
            #no tanh function becuase we're going for a dB prediction
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x


class cnn_model_more_fc(nn.Module):  #REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_more_fc, self).__init__()
        self.conv = nn.Sequential(
            # 5x5
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            # 3x3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),

            # 3x3, second part
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 10, 1000),  # input size is 16x16 after conv layers
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            nn.Linear(1000, 750),
            nn.BatchNorm1d(750),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            nn.Linear(750, 500),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            nn.Linear(500, 200),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),
            nn.Linear(200, 3 * output_w),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x


class cnn_model_2(nn.Module):  # REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 10, 1000),  # input size is 16x18 after conv layers
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            nn.Linear(500, 3 * output_w),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1  # Residual connection
        x3 = self.conv3(x2) + x2  # Residual connection
        x4 = self.conv4(x3)
        x5 = self.conv5(x4) + x4  # Residual connection
        x6 = self.conv6(x5) + x5  # Residual connection
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8) + x8  # Residual connection
        x10 = self.conv10(x9) + x9  # Residual connection
        x11 = self.conv11(x10) + x10  # Residual connection
        x12 = self.conv12(x11) + x11  # Residual connection

        x = x12.view(x12.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class cnn_model_3(nn.Module):  # REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(cnn_model_3, self).__init__()
        self.conv = nn.Sequential(
            # 7x7
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05),

            # 5x5
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05),

            # 3x3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.05),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 10, 1000),  # input size is 16x18 after conv layers, 8x10 if you work on the 8x8
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            nn.Linear(500, 3 * output_w),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x
