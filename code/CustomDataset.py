import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, augmented_data):
        self.augmented_data = augmented_data

    def __len__(self):
        # Triple the dataset length for each item being seen three times (original, flipud, fliplr)
        return len(self.augmented_data) * 3

    def __getitem__(self, idx):
        original_idx = idx // 3
        input_matrix, target_matrix = self.augmented_data[original_idx]

        # Create a copy to avoid modifying the original data
        input_matrix = np.array(input_matrix)
        target_matrix = np.array(target_matrix)

        # Apply flipping based on the index
        if idx % 3 == 1:
            input_matrix = np.flipud(input_matrix).copy()  # Vertical flip\
        elif idx % 3 == 2:
            input_matrix = np.fliplr(input_matrix).copy()  # Horizontal flip
            # Assuming target_matrix is structured as [dB(S11), dB(S12), dB(S22)]
            # Swap S11 and S22 components (only)
            target_matrix[[0, 2]] = target_matrix[[2, 0]]

        # Convert numpy arrays to PyTorch tensors
        input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        target_tensor = torch.tensor(target_matrix, dtype=torch.float32)

        return input_tensor, target_tensor
