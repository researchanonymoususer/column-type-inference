import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class BaseNeuralNetworkModel(ABC, nn.Module):
    def __init__(self, criterion):
        super(BaseNeuralNetworkModel, self).__init__()
        self.criterion = criterion
        self.metrics = {}
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.best_model_state = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the network. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def prep_data_loader(self, X_train, y_train, X_val, y_val, X_test, y_test,
                        batch_size=32):
        pass

    def train_epochs(self, optimizer, scheduler, epochs=50, device: str = 'cpu',
                     patience=10, min_delta=0.001, monitor='val_acc'):
        """
        Train the model on training data with early stopping.
        :param optimizer: torch.optim object
        :param epochs: Number of epochs
        :param device: Device to run training on ('cpu' or 'cuda')
        :param patience: Number of epochs to wait for improvement before stopping
        :param min_delta: Minimum change to qualify as improvement
        :param monitor: Metric to monitor ('val_acc' or 'val_gmean')
        """

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        val_gmeans = []
        best_val_metric = 0
        self.best_model_state = None
        val_gmeans_weighted = []
        val_gmeans_multiclass = []

        # Early stopping variables
        epochs_no_improve = 0
        early_stop = False

        for epoch in range(epochs):
            if early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            self.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for batch_data in self.train_loader:
                # Handle different input formats
                if len(batch_data) == 3:  # (input1, input2, labels)
                    input1, input2, labels = batch_data
                    input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
                    inputs = (input1, input2)
                elif len(batch_data) == 2:  # (inputs, labels)
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = (inputs,)
                else:
                    raise ValueError("Unexpected batch format")

                optimizer.zero_grad()
                outputs = self.forward(*inputs)
                loss = self.criterion(outputs, labels.squeeze(1))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels.squeeze()).sum().item()

            # Validate
            self.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for batch_data in self.val_loader:
                    # Handle different input formats
                    if len(batch_data) == 3:  # (input1, input2, labels)
                        input1, input2, labels = batch_data
                        input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
                        inputs = (input1, input2)
                    elif len(batch_data) == 2:  # (inputs, labels)
                        inputs, labels = batch_data
                        inputs, labels = inputs.to(device), labels.to(device)
                        inputs = (inputs,)
                    else:
                        raise ValueError("Unexpected batch format")

                    outputs = self.forward(*inputs)
                    loss = self.criterion(outputs, labels.squeeze(1))

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels.squeeze()).sum().item()

                    # Store predictions for G-Mean calculation
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.squeeze().cpu().numpy())

            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            # Calculate G-Mean — both versions
            val_gmean_weighted = geometric_mean_score(
                np.array(all_val_labels).ravel(),
                np.array(all_val_preds).ravel(),
                average='weighted'
            )
            val_gmean_multiclass = geometric_mean_score(
                np.array(all_val_labels).ravel(),
                np.array(all_val_preds).ravel(),
                average='multiclass'
            )

            val_gmeans_weighted.append(val_gmean_weighted)
            val_gmeans_multiclass.append(val_gmean_multiclass)

            # Monitor weighted for stable early stopping
            current_val_metric = val_gmean_weighted if monitor == 'val_gmean' else val_acc

            train_losses.append(train_loss / len(self.train_loader))
            val_losses.append(val_loss / len(self.val_loader))
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            scheduler.step()

            # Early stopping logic
            if current_val_metric > best_val_metric + min_delta:
                best_val_metric = current_val_metric
                self.best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
                metric_name = "G-Mean" if monitor == 'val_gmean' else "accuracy"
                print(f"Epoch [{epoch + 1}/{epochs}] - New best validation {metric_name}: {current_val_metric:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    early_stop = True

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] - "
                      f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%, Val G-Mean(weighted): {val_gmean_weighted:.4f} | "
                      f"No improve: {epochs_no_improve}/{patience}")

        self.metrics['train_losses'] = train_losses
        self.metrics['train_accuracies'] = train_accs
        self.metrics['val_losses'] = val_losses
        self.metrics['val_accuracies'] = val_accs

        self.metrics['val_gmeans'] = val_gmeans_multiclass  # multiclass, for plots
        self.metrics['val_gmeans_monitor'] = val_gmean_weighted  # weighted, used for selection
        self.metrics['best_val_metric'] = best_val_metric  # weighted, used for selection

        self.metrics['stopped_epoch'] = epoch + 1
        self.metrics['monitor_metric'] = monitor

    def evaluate(self, X_test_column, device: str = 'cpu'):
        """
        Evaluate the model on test data.
        :param test_loader: DataLoader containing test data
        :param device: Device to run evaluation on ('cpu' or 'cuda')
        """
        self.load_state_dict(self.best_model_state)
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data in self.test_loader:
                # Handle different input formats
                if len(batch_data) == 3:  # (input1, input2, labels)
                    input1, input2, labels = batch_data
                    input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
                    inputs = (input1, input2)
                elif len(batch_data) == 2:  # (inputs, labels)
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = (inputs,)
                else:
                    raise ValueError("Unexpected batch format")

                # Forward pass
                outputs = self.forward(*inputs)
                loss = self.criterion(outputs, labels.squeeze(1))

                # Calculate metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

                # Store predictions and labels for sklearn metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        all_preds = all_preds.ravel()
        all_labels = all_labels.ravel()

        mis_idx = np.where(all_labels != all_preds)[0]
        print(f"\nMisclassified indices: {mis_idx}")
        print(f"Total misclassifications: {len(mis_idx)}")

        if len(mis_idx) > 0:
            X_test_column_reset = X_test_column.reset_index(drop=True)
            misclassified_df = pd.DataFrame({
                'Feature_Value': X_test_column_reset.iloc[mis_idx].values,
                'Actual': all_labels[mis_idx],
                'Predicted': all_preds[mis_idx],
            })
            print("\nMisclassified samples with predictions:")
            print(misclassified_df)

        # Calculate metrics
        self.metrics['test_loss'] = total_loss / len(self.test_loader)
        self.metrics['test_accuracy'] = 100 * correct / total
        self.metrics['test_precision'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        self.metrics['test_recall'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        self.metrics['test_f1score'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        self.metrics['test_gmean'] = geometric_mean_score(all_labels.ravel(), all_preds.ravel(), average='multiclass')
        self.metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)

    def evaluate_ensemble(self, rf_model, X_test, alpha=0.3, device: str = 'cpu', split='test'):
        """
        Evaluate ensemble of NN + Random Forest on test data.
        :param rf_model: trained TreeModel (Random Forest) instance
        :param X_test: test features for RF (numpy array or DataFrame)
        :param alpha: weight for NN predictions (1-alpha for RF)
        :param device: device for NN inference
        :param split: 'test' or 'val' — which loader to use for NN inference
        """
        self.load_state_dict(self.best_model_state)
        self.eval()

        # Choose loader based on split
        loader = self.val_loader if split == 'val' else self.test_loader

        all_nn_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_data in loader:
                if len(batch_data) == 3:
                    input1, input2, labels = batch_data
                    input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
                    inputs = (input1, input2)
                elif len(batch_data) == 2:
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = (inputs,)
                else:
                    raise ValueError("Unexpected batch format")

                outputs = self.forward(*inputs)
                nn_probs = torch.softmax(outputs, dim=1)
                all_nn_probs.extend(nn_probs.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())

        all_nn_probs = np.array(all_nn_probs)
        all_labels = np.array(all_labels).ravel()

        rf_probs = rf_model.predict_proba(X_test)

        # Sanity check
        assert all_nn_probs.shape[0] == rf_probs.shape[0], (
            f"Shape mismatch: NN={all_nn_probs.shape[0]} samples, RF={rf_probs.shape[0]} samples. "
            f"Ensure X_test matches the '{split}' split."
        )

        ensemble_probs = alpha * all_nn_probs + (1 - alpha) * rf_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        correct = (ensemble_preds == all_labels).sum()
        total = len(all_labels)

        self.metrics['ensemble_alpha'] = alpha
        self.metrics['ensemble_test_accuracy'] = 100 * correct / total
        self.metrics['ensemble_test_precision'] = precision_score(all_labels, ensemble_preds, average='weighted',
                                                                  zero_division=0) * 100
        self.metrics['ensemble_test_recall'] = recall_score(all_labels, ensemble_preds, average='weighted',
                                                            zero_division=0) * 100
        self.metrics['ensemble_test_f1score'] = f1_score(all_labels, ensemble_preds, average='weighted',
                                                         zero_division=0)
        self.metrics['ensemble_test_gmean'] = geometric_mean_score(all_labels, ensemble_preds, average='multiclass')
        self.metrics['ensemble_confusion_matrix'] = confusion_matrix(all_labels, ensemble_preds)
