import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model.neural_network_model import BaseNeuralNetworkModel

class ModularNeuralNetwork(BaseNeuralNetworkModel):
    """
    Feed Forward Neural Network with two subnetworks
    Architecture:
    - Subnetwork 1: processes input of size 95
    - Subnetwork 2: processes input of size 200
    - Merged features are combined and passed through final layers
    - Output: probabilities of 4 classes
    """
    def __init__(self, criterion):
        super(ModularNeuralNetwork, self).__init__(criterion)
        # Subnetwork 1 (Input size: 95)
        self.subnet1_fc1 = nn.Linear(95, 128)
        self.subnet1_bn1 = nn.BatchNorm1d(128)
        self.subnet1_dropout1 = nn.Dropout(0.5)
        self.subnet1_fc2 = nn.Linear(128, 64)
        self.subnet1_bn2 = nn.BatchNorm1d(64)
        self.subnet1_dropout2 = nn.Dropout(0.3)

        # Subnetwork 2 (Input size: 200)
        self.subnet2_fc1 = nn.Linear(200, 256)
        self.subnet2_bn1 = nn.BatchNorm1d(256)
        self.subnet2_dropout1 = nn.Dropout(0.5)
        self.subnet2_fc2 = nn.Linear(256, 128)
        self.subnet2_bn2 = nn.BatchNorm1d(128)
        self.subnet2_dropout2 = nn.Dropout(0.3)

        # Merged layers (64 + 64 = 128 combined features)
        self.merged_fc1 = nn.Linear(192, 64)
        self.merged_bn1 = nn.BatchNorm1d(64)
        self.merged_dropout1 = nn.Dropout(0.3)
        # Output layer
        self.output_fc = nn.Linear(64, 4)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def forward(self, input1, input2):
        # Subnetwork 1 forward pass
        x1 = self.subnet1_fc1(input1)
        x1 = self.subnet1_bn1(x1)
        x1 = F.relu(x1)
        x1 = self.subnet1_dropout1(x1)
        x1 = self.subnet1_fc2(x1)
        x1 = self.subnet1_bn2(x1)
        x1 = F.relu(x1)
        x1 = self.subnet1_dropout2(x1)
        # Subnetwork 2 forward pass
        x2 = self.subnet2_fc1(input2)
        x2 = self.subnet2_bn1(x2)
        x2 = F.relu(x2)
        x2 = self.subnet2_dropout1(x2)
        x2 = self.subnet2_fc2(x2)
        x2 = self.subnet2_bn2(x2)
        x2 = F.relu(x2)
        x2 = self.subnet2_dropout2(x2)

        combined = torch.cat([x1, x2], dim=1)

        # Final layers
        x = self.merged_fc1(combined)
        x = self.merged_bn1(x)
        x = F.relu(x)
        x = self.merged_dropout1(x)

        # Output layer
        x = self.output_fc(x)

        return x

    def prep_data_loader(self, X_train, y_train, X_val, y_val, X_test, y_test,
                        batch_size=32):
        # Split features for two subnetworks
        X_train_1, X_train_2 = X_train.iloc[:, :95].to_numpy(dtype=np.float32), X_train.iloc[:, 95:].to_numpy(
            dtype=np.float32)
        X_val_1, X_val_2 = X_val.iloc[:, :95].to_numpy(dtype=np.float32), X_val.iloc[:, 95:].to_numpy(dtype=np.float32)
        X_test_1, X_test_2 = X_test.iloc[:, :95].to_numpy(dtype=np.float32), X_test.iloc[:, 95:].to_numpy(
            dtype=np.float32)
        y_train = y_train.to_numpy(dtype=np.int64)
        y_val = y_val.to_numpy(dtype=np.int64)
        y_test = y_test.to_numpy(dtype=np.int64)

        # Convert to tensors
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_1),
            torch.from_numpy(X_train_2),
            torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_1),
            torch.from_numpy(X_val_2),
            torch.from_numpy(y_val)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test_1),
            torch.from_numpy(X_test_2),
            torch.from_numpy(y_test)
        )

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
