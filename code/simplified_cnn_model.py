import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow.raft import ResidualBlock


def dB(x_re, x_im):
    return 20 * torch.log10(torch.sqrt(x_re**2 + x_im**2) + 1e-6)  # Added epsilon for numerical stability

def Combined_H_Loss(outputs, targets, output_w=70, alpha=0.9, delta=3.0):
    loss_total = 0
    for i in range(3):
        slice_re = slice(i * output_w, (i + 1) * output_w)
        slice_im = slice((i + 3) * output_w, (i + 4) * output_w)
        predicted_re = outputs[:, slice_re]
        predicted_im = outputs[:, slice_im]
        actual_re = targets[:, slice_re]
        actual_im = targets[:, slice_im]
        predicted_db = dB(predicted_re, predicted_im)
        actual_db = dB(actual_re, actual_im)
        loss_db = F.huber_loss(predicted_db, actual_db, delta=delta)
        loss_total += loss_db
    loss_db_avg = loss_total / 3
    loss_mse = F.mse_loss(outputs, targets)
    loss_combined = alpha * loss_mse + (1 - alpha) * loss_db_avg
    return loss_combined



def smoothness_penalty(outputs, weight=1.0):
    # Calculate the difference between adjacent frequency points
    diffs = outputs[:, 1:] - outputs[:, :-1]
    return weight * torch.mean(torch.pow(diffs, 2))


def Combined_H_SMOOTH_Loss(outputs, targets, output_w=70, alpha=0.9, beta=0.8, delta=3.0):
    loss_total = 0
    for i in range(3):  # For S11, S21, S22 each with real and imaginary parts
        slice_re = slice(i * output_w, (i + 1) * output_w)
        slice_im = slice((i + 3) * output_w, (i + 4) * output_w)
        predicted_re = outputs[:, slice_re]
        predicted_im = outputs[:, slice_im]
        actual_re = targets[:, slice_re]
        actual_im = targets[:, slice_im]

        # Convert real and imaginary parts to dB
        predicted_db = dB(predicted_re, predicted_im)
        actual_db = dB(actual_re, actual_im)

        # Huber loss for dB values
        loss_db = F.huber_loss(predicted_db, actual_db, delta=delta)
        loss_total += loss_db

    # Average dB loss across all sets of parameters
    loss_db_avg = loss_total / 3

    # Mean squared error for all outputs
    loss_mse = F.mse_loss(outputs, targets)

    # Combined loss with dB and MSE
    combined_loss = alpha * loss_mse + (1 - alpha) * loss_db_avg

    # Adding smoothness penalty to encourage smoother transitions between frequency points
    smooth_penalty = smoothness_penalty(outputs, weight=beta)

    # Final combined loss
    total_loss = combined_loss + smooth_penalty

    return total_loss

# Regular one is 1 FC and 7 CONV

import torch.nn as nn


class SimplifiedCNN(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, output_w * 3),  # Assume output_w is the number of frequency points
            # No activation here, output linear values suitable for dB scale
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariedKernelCNN(nn.Module):
    def __init__(self, output_w):
        super(VariedKernelCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),  # Larger filter, maintain dimensions
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),  # Continue with larger filter
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),  # Even larger filter
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),  # Consistent with previous layer
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=9, stride=1, padding=4),  # Largest filter in the series
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4),  # Same size to maintain feature scale
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # Pool to a 1x1 feature map
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(128, output_w * 3),  # Outputs a vector of length output_w * 6
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class SimplifiedCNN_just_trying_something(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN_just_trying_something, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, output_w * 3),  # Adjust the input features to match the output of last conv layer
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class SimplifiedCNN_3fc(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN_3fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class SimplifiedCNN_5fc(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN_5fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class SimplifiedCNN_6fc(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN_6fc, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class SimplifiedCNN_8fc_6conv(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN_8fc_6conv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.02),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.02),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.03),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x
class SimplifiedCNN_11conv(nn.Module):  # default simplified CNN has 7 layers
    def __init__(self, output_w):
        super(SimplifiedCNN_11conv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
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
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(128, output_w * 3),  # Adjust the input features to match the output of last conv layer
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class expressiveCNN_2(nn.Module):
    def __init__(self, output_w):
        super(expressiveCNN_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            ResidualBlock(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)
        return x

class expressiveCNN_3(nn.Module):
    def __init__(self, output_w):
        super(expressiveCNN_3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(alpha=1.0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=1.0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(alpha=1.0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(alpha=1.0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(alpha=1.0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(alpha=1.0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 500),
            nn.ELU(alpha=1.0),
            nn.Linear(500, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)
        return x

class expressiveCNN_1(nn.Module):
    def __init__(self, output_w):
        super(expressiveCNN_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 500),
            nn.ReLU(),
            nn.Linear(500, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)
        return x


class SimplifiedCNN_2fc_8conv(nn.Module):  # default simplified CNN has 7 layers
    def __init__(self, output_w):
        super(SimplifiedCNN_2fc_8conv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the network input size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(128, 500),
            nn.LeakyReLU(0.010),  # Adjust the input features to match the output of last conv layer
            nn.Linear(500, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class SimplifiedCNN_5fc_8conv_v2(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN_5fc_8conv_v2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.ELU(),
            nn.Linear(1000, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class SimplifiedCNN_5fc_8conv_v3(nn.Module):
    def __init__(self, output_w):
        super(SimplifiedCNN_5fc_8conv_v3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 300),
            nn.PReLU(),
            nn.Linear(300, 300),
            nn.PReLU(),
            nn.Linear(300, 300),
            nn.PReLU(),
            nn.Linear(300, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x


class NewCNN(nn.Module):
    def __init__(self, output_w):
        super(NewCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.ELU(),
            nn.Linear(1000, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class EnhancedNewCNN(nn.Module):
    def __init__(self, output_w):
        super(EnhancedNewCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x
class NewCNN_2(nn.Module):
    def __init__(self, output_w):
        super(NewCNN_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w * 3),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

class NewCNN_3(nn.Module):
    def __init__(self, output_w):
        super(NewCNN_3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)
        return x
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = self.mish(out)
        return out

class EnhancedCNN(nn.Module):
    def __init__(self, output_w):
        super(EnhancedCNN, self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Mish()
        )
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1500),
            Mish(),
            nn.Linear(1500, 2000),
            Mish(),
            nn.Linear(2000, 1000),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(1000, output_w * 3)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)
        return x

class NewCNN_db_in_forward(nn.Module):
    def __init__(self, output_w):
        super(NewCNN_db_in_forward, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        # Convert to dB, adding a small constant epsilon to avoid log(0)
        epsilon = 1e-6
        x = 10 * torch.log10(torch.abs(x) + epsilon)
        return x

class Triple_CNN(nn.Module):
    def __init__(self, output_w):
        super(Triple_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w),
        )
        self.features_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier_2 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w),
        )
        self.features_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier_3 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w),
        )

    def forward(self, x):
        y = x.clone()
        z = x.clone()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        y = self.features_2(y)
        y = y.view(y.size(0), -1)
        y = self.classifier_2(y)

        z = self.features_3(z)
        z = z.view(z.size(0), -1)
        z = self.classifier_3(z)

        combined_output = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
        return combined_output

import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out

class NewCNN_2_Enhanced(nn.Module):
    def __init__(self, output_w):
        super(NewCNN_2_Enhanced, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=0),  # Larger kernel size
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  # Adding dropout
            ResidualBlock(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  # Adding dropout
            ResidualBlock(128, 256, stride=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 1000),
            nn.ELU(),
            nn.Linear(1000, 1500),
            nn.ELU(),
            nn.Linear(1500, 2000),
            nn.ELU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.01),
            nn.Linear(1000, output_w * 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x

import torch
import torch.nn as nn

class StudyCNN(nn.Module):
    def __init__(self, output_w):
        super(StudyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(12, 12), stride=1, padding=5),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(10, 10), stride=1, padding=4),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(8, 8), stride=1, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(6, 6), stride=1, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(5, 5), stride=1, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(5, 5), stride=1, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 500),
            nn.LeakyReLU(0.01),
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),
            nn.Linear(500, 500),
            nn.LeakyReLU(0.01),
            nn.Linear(500, 3*output_w),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x


class residual_cnn_model(nn.Module):  # REMEMBER - INPUT IS ACTUALLY 16X18!
    def __init__(self, output_w):
        super(residual_cnn_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 16 * 18, 1000),  # input size is 16x18 after conv layers
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
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1  # Residual connection
        x3 = self.conv3(x2) + x2  # Residual connection
        x4 = self.conv4(x3)
        x5 = self.conv5(x4) + x4  # Residual connection
        x6 = self.conv6(x5) + x5  # Residual connection
        x7 = self.conv7(x6) + x6  # Residual connection
        x8 = self.conv8(x7)
        x9 = self.conv9(x8) + x8  # Residual connection
        x10 = self.conv10(x9) + x9  # Residual connection
        x11 = self.conv11(x10) + x10  # Residual connection
        x12 = self.conv12(x11) + x11  # Residual connection
        x13 = self.conv13(x12) + x12  # Residual connection
        x14 = self.conv14(x13) + x13  # Residual connection
        x15 = self.conv15(x14) + x14  # Residual connection
        x16 = self.conv16(x15) + x15  # Residual connection

        x = x16.view(x16.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = x.view(x.size(0), 3, -1)  # Reshape to (batch_size, 3, output_w)
        return x