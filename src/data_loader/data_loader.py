import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Custom dataset class for handling image-label pairs
class CustomDataset(Dataset):
    def __init__(self, images, labels, device):
        self.images = images
        self.labels = torch.FloatTensor(labels)
        self.device = device
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float).to(self.device)
        label = self.labels[idx].to(self.device)
        return image, label


def load_data(processed_data_dir, device, batch_size=32):
    # Load processed data from files
    X_train_temp = np.load(os.path.join(processed_data_dir, 'X_train_temp.npy')).reshape(-1, 224, 224, 3).transpose(0, 3, 1, 2)
    y_train_temp = np.load(os.path.join(processed_data_dir, 'y_train_temp.npy'))

    X_val = np.load(os.path.join(processed_data_dir, 'X_val.npy')).reshape(-1, 224, 224, 3).transpose(0, 3, 1, 2)
    y_val = np.load(os.path.join(processed_data_dir, 'y_val.npy'))

    X_test = np.load(os.path.join(processed_data_dir, 'X_test.npy')).reshape(-1, 224, 224, 3).transpose(0, 3, 1, 2)
    y_test = np.load(os.path.join(processed_data_dir, 'y_test.npy'))

    # Create datasets
    train_dataset = CustomDataset(X_train_temp, y_train_temp, device)
    val_dataset = CustomDataset(X_val, y_val, device)
    test_dataset = CustomDataset(X_test, y_test, device)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader