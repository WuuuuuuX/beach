import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class BeachProfileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset class for beach equilibrium profile data.

        Args:
            root_dir (str): Directory containing class folders (0-4) with Excel files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Define the features we want to extract
        self.features = [
            "x", "dryBeach", "podu", "Azimuth", "Length",
            "averageParticleSize", "averageParticleSize2", "sortingFactor",
            "Skewness", "peakState", "meanAnnualHighTidalRange",
            "meanAnnualTidalRange", "waveDirection", "frequency",
            "averageWaveHeight", "Periodicity", "RTR", "Omega",
            "HightideSedimentsettlingvelocity", "Hb", "Hd",
            "averageAnnualPeriodicity"
        ]
        self.target = "y"

        # Load all data files
        for class_idx in range(5):  # 5 classes: 0-4
            class_dir = os.path.join(root_dir, str(class_idx))
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith('.xlsx'):
                        file_path = os.path.join(class_dir, filename)
                        try:
                            data = pd.read_excel(file_path)

                            # Check if all required features and target exist
                            if all(feat in data.columns for feat in self.features) and self.target in data.columns:
                                # Store features and targets row by row instead of as whole arrays
                                # This ensures we don't have shape mismatches
                                for i in range(len(data)):
                                    feature_row = data[self.features].iloc[i].values
                                    target_value = data[self.target].iloc[i]

                                    self.samples.append({
                                        'features': feature_row,
                                        'target': target_value,
                                        'class': class_idx,
                                        'file_path': file_path
                                    })
                            else:
                                missing_cols = [col for col in self.features + [self.target] if col not in data.columns]
                                print(f"Warning: Missing columns {missing_cols} in file {file_path}")
                        except Exception as e:
                            print(f"Error loading file {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]

        # Convert to tensors
        features = torch.tensor(sample['features'], dtype=torch.float32)
        target = torch.tensor(sample['target'], dtype=torch.float32)
        class_label = torch.tensor(sample['class'], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return {
            'features': features,
            'target': target,
            'class': class_label
        }

    def get_data_by_class(self, class_idx):
        """Get all data for a specific class"""
        class_data = []
        class_targets = []

        for sample in self.samples:
            if sample['class'] == class_idx:
                class_data.append(sample['features'])
                class_targets.append(sample['target'])

        if not class_data:
            return None, None

        return torch.tensor(np.vstack(class_data), dtype=torch.float32), torch.tensor(class_targets,
                                                                                      dtype=torch.float32)


def get_dataloaders(train_dir, val_dir, batch_size, num_workers=4):
    """
    Create DataLoaders for both training and validation datasets

    Args:
        train_dir (str): Directory for training data
        val_dir (str): Directory for validation data
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker threads for loading data

    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    train_dataset = BeachProfileDataset(root_dir=train_dir) if train_dir else None
    val_dataset = BeachProfileDataset(root_dir=val_dir) if val_dir else None

    train_loader = None
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return train_loader, val_loader, train_dataset, val_dataset