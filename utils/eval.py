import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset


class Eval:
    def __init__(
        self,
        model: LSTMNetKAN,
        train_loader: DataLoader,
        test_loader: DataLoader,
        scaler: MinMaxScaler,
        dataset: WetlandDataset,
        eval_folder: str,
        model_folder: str,
        device: torch.device,
        lats: np.ndarray,
        lons: np.ndarray,
    ):
        """
        Initialize the evaluation process with model, data loaders, scaler, and dataset.

        Args:
            model (LSTMNetKAN): The trained model to evaluate.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            scaler (dict[str, MinMaxScaler]): Dictionary of scalers for inverse transforming predictions.
            dataset (WetlandDataset_Loc): The dataset used for evaluation.
            eval_folder (str): Folder path to save evaluation results.
            model_folder (str): Folder path where the trained model is saved.
            device (torch.device): Device to run the evaluation on.
            lats (np.ndarray): Array of latitude indices for the locations.
            lons (np.ndarray): Array of longitude indices for the locations.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.dataset = dataset
        self.eval_folder = eval_folder
        self.model_folder = model_folder
        self.lats = lats
        self.lons = lons
        self.device = device

    def run(self):
        if not os.path.exists(self.eval_folder):
            os.makedirs(self.eval_folder)

        for lat in self.lats:
            for lon in self.lons:
                model_path = os.path.join(self.model_folder, f"{lat}_{lon}.pth")
                if not os.path.exists(model_path):
                    print(f"Model for location ({lat}, {lon}) not found. Skipping.")
                    continue
                print(f"Processing location ({lat}, {lon})...")

                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.eval()

                eval_path = os.path.join(self.eval_folder, f"{lat}_{lon}.npy")
                np.save(
                    eval_path,
                    {
                        "Y_inv": self.Y_inv,
                        "metrics": self.metrics,
                    },
                )

    @staticmethod
    def sMAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Symmetric Mean Absolute Percentage Error (sMAPE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The sMAPE value.
        """
        return (
            (100.0 / len(y_true))
            * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
            / 2
        )

    @staticmethod
    def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The NSE value.
        """
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (numerator / denominator)

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled predictions to original scale.

        Args:
            y_scaled (np.ndarray): Scaled predictions.

        Returns:
            np.ndarray: Predictions in original scale.
        """
        y = np.concatenate(y_scaled).reshape(-1, 1)
        y_inv = self.scaler.inverse_transform(y)
        y_inv[y_inv < 0.0003] = 0.0  # Set small negative values to zero
        return y_inv

    def eval(self):
        self.model.eval()
        Y = {
            "y_pred_train": [],
            "y_true_train": [],
            "y_pred_test": [],
            "y_true_test": [],
        }

        train_y_trues = []
        for inputs, targets in self.train_loader:
            train_y_trues.append(targets.cpu().numpy().reshape(-1))
        train_y_trues = np.concatenate(train_y_trues)

        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()

                h = self.model.init_hidden(inputs.size(0))
                outputs, h = self.model(inputs, h)
                Y["y_pred_train"].append(outputs.cpu().numpy().reshape(-1))
                Y["y_true_train"].append(targets.cpu().numpy().reshape(-1))

            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()

                h = self.model.init_hidden(inputs.size(0))
                outputs, h = self.model(inputs, h)
                Y["y_pred_test"].append(outputs.cpu().numpy().reshape(-1))
                Y["y_true_test"].append(targets.cpu().numpy().reshape(-1))

        Y_inv = {name: self.inverse(Y[name]) for name in Y}

        self.Y_inv = Y_inv
        self.metrics = {
            "train": {
                "sMAPE": self.sMAPE(Y_inv["y_true_train"], Y_inv["y_pred_train"]),
                "NSE": self.nse(Y_inv["y_true_train"], Y_inv["y_pred_train"]),
                "R2": r2_score(Y_inv["y_true_train"], Y_inv["y_pred_train"]),
                "RMSE": np.sqrt(
                    mean_squared_error(Y_inv["y_true_train"], Y_inv["y_pred_train"])
                ),
            },
            "test": {
                "sMAPE": self.sMAPE(Y_inv["y_true_test"], Y_inv["y_pred_test"]),
                "NSE": self.nse(Y_inv["y_true_test"], Y_inv["y_pred_test"]),
                "R2": r2_score(Y_inv["y_true_test"], Y_inv["y_pred_test"]),
                "RMSE": np.sqrt(
                    mean_squared_error(Y_inv["y_true_test"], Y_inv["y_pred_test"])
                ),
            },
        }
