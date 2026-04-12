# Generic Model Class for Decision Tree and Random Forest

import numpy as np
import pandas as pd

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class TreeModel:
    def __init__(self, name, model):
        """
        Initialize the TreeModel class
        :param name: name of the model
        :param model: sklearn's DecisionTreeClassifier or RandomForestClassifier
        """
        self.model_name = name
        self.model = model
        self.metrics = {}

    def train(self, X, y):
        """
        Train the Tree Classifier
        :param X: training data
        :param y: training labels
        """
        print(f"Training {self.model_name}...")
        self.model.fit(X, y)

    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, X_test_column):
        """
        Generate evaluation metrics for the classifier and
        store it in metrics dictionary
        :param X_train: training data
        :param y_train: training labels
        :param X_val: validation data
        :param y_val: validation labels
        :param X_test: test data
        :param y_test: test labels
        :param X_test_column: original column names for the test set (for error analysis)
        """
        train_preds = self.predict(X_train)
        val_preds = self.predict(X_val)
        test_preds = self.predict(X_test)

        y_test_arr = y_test.values.ravel()
        test_preds = np.asarray(test_preds).ravel()

        mis_idx = np.where(y_test_arr != test_preds)[0]
        print(f"\n{self.model_name} - Misclassified indices: {mis_idx}")
        print(f"Total misclassifications: {len(mis_idx)}")

        if len(mis_idx) > 0:
            X_test_column_reset = X_test_column.reset_index(drop=True)
            misclassified_df = pd.DataFrame({
                'Feature_Value': X_test_column_reset.iloc[mis_idx].values,
                'Actual':        y_test_arr[mis_idx],
                'Predicted':     test_preds[mis_idx],
            })
            print("\nMisclassified samples with predictions:")
            print(misclassified_df)

        self.metrics['train_accuracy']       = accuracy_score(y_train, train_preds) * 100
        self.metrics['validation_accuracy']  = accuracy_score(y_val, val_preds) * 100
        self.metrics['validation_precision'] = precision_score(y_val, val_preds, average='weighted', zero_division=0) * 100
        self.metrics['validation_recall']    = recall_score(y_val, val_preds, average='weighted', zero_division=0) * 100
        self.metrics['validation_f1score']   = f1_score(y_val, val_preds, average='weighted', zero_division=0)
        self.metrics['validation_gmean']     = geometric_mean_score(y_val, val_preds, average='multiclass')

        self.metrics['test_accuracy']  = accuracy_score(y_test, test_preds) * 100
        self.metrics['test_precision'] = precision_score(y_test, test_preds, average='weighted', zero_division=0) * 100
        self.metrics['test_recall']    = recall_score(y_test, test_preds, average='weighted', zero_division=0) * 100
        self.metrics['test_f1score']   = f1_score(y_test, test_preds, average='weighted', zero_division=0)
        self.metrics['test_gmean']     = geometric_mean_score(y_test, test_preds, average='multiclass')

        self.metrics['confusion_matrix'] = confusion_matrix(y_test, test_preds)

    def predict(self, X):
        """
        Predict labels for given data
        :param X: test data
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities of labels for given data
        :param X: test data
        """
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names):
        """
        Returns a dict of {feature_name: importance} sorted descending.
        :param feature_names: list of feature names
        """
        importances = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))

        sorted_importance = sorted(importance_dict.items(),
                                   key=lambda x: x[1],
                                   reverse=True)

        print("\nTop 15 Most Important Features:")
        for feature, importance in sorted_importance[:15]:
            print(f"  {feature}: {importance:.4f}")

        return importance_dict
