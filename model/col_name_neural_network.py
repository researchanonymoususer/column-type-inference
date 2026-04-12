import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model.neural_network_model import BaseNeuralNetworkModel

class ColNameNeuralNetwork(BaseNeuralNetworkModel):
    """
    Second subnetwork - processes input of size 200
    Architecture:
    - Input: 200 features
    - Hidden layer 1: 256 units with BatchNorm and Dropout
    - Hidden layer 2: 128 units with BatchNorm and Dropout
    - Output: 4 classes
    """

    def __init__(self, criterion):
        super(ColNameNeuralNetwork, self).__init__(criterion)

        # Layers
        self.fc1 = nn.Linear(200, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        # Output layer
        self.output_fc = nn.Linear(128, 4)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer with softmax
        x = self.output_fc(x)
        output = F.softmax(x, dim=1)

        return output

    def prep_data_loader(self, X_train, y_train, X_val, y_val, X_test, y_test,
                         batch_size=32):
        # Use only last 200 features (columns 95 onwards)
        X_train_2 = X_train.iloc[:, 95:].to_numpy(dtype=np.float32)
        X_val_2 = X_val.iloc[:, 95:].to_numpy(dtype=np.float32)
        X_test_2 = X_test.iloc[:, 95:].to_numpy(dtype=np.float32)

        y_train = y_train.to_numpy(dtype=np.int64)
        y_val = y_val.to_numpy(dtype=np.int64)
        y_test = y_test.to_numpy(dtype=np.int64)

        # Convert to tensors
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_2),
            torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_2),
            torch.from_numpy(y_val)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test_2),
            torch.from_numpy(y_test)
        )

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
