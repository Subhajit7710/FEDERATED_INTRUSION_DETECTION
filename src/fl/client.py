
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import flwr as fl


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path("/content/IDSPRESERVING")
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "scaled"
CLIENT_DIR = PROJECT_ROOT / "data" / "processed" / "clients"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import IDSModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 1024
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# If set, only this many samples are used.
# Example:
# SMOKE_TEST_SAMPLES=5000
SMOKE_TEST_SAMPLES = int(
    os.environ.get("SMOKE_TEST_SAMPLES", "0")
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class ClientDataset(Dataset):

    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        # Make a writable NumPy copy to avoid
        # the mmap -> PyTorch warning.
        x = np.array(
            self.X[real_idx],
            dtype=np.float32,
            copy=True
        )

        y = np.float32(
            self.y[real_idx]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(
                y,
                dtype=torch.float32
            )
        )


# --------------------------------------------------
# Flower Client
# --------------------------------------------------

class IDSClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id

        print(
            f"\nInitializing Client {client_id}"
        )

        print(
            f"Device: {DEVICE}"
        )

        # Memory-mapped training data
        self.X_train = np.load(
            DATA_DIR / "X_train.npy",
            mmap_mode="r"
        )

        self.y_train = np.load(
            DATA_DIR / "y_train.npy",
            mmap_mode="r"
        )

        # Client partition
        self.client_indices = np.load(
            CLIENT_DIR /
            f"client_{client_id}_indices.npy"
        )

        # Optional smoke-test limit
        if SMOKE_TEST_SAMPLES > 0:

            self.client_indices = (
                self.client_indices[
                    :SMOKE_TEST_SAMPLES
                ]
            )

            print(
                f"SMOKE TEST: using "
                f"{len(self.client_indices):,} samples"
            )

        print(
            f"Client {client_id} samples: "
            f"{len(self.client_indices):,}"
        )

        # Dataset
        dataset = ClientDataset(
            self.X_train,
            self.y_train,
            self.client_indices
        )

        self.loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        # Model
        self.model = IDSModel(
            input_size=self.X_train.shape[1]
        ).to(DEVICE)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )


    # --------------------------------------------------
    # Get parameters
    # --------------------------------------------------

    def get_parameters(self, config):

        return [
            value.detach()
            .cpu()
            .numpy()
            for value in self.model.state_dict().values()
        ]


    # --------------------------------------------------
    # Set parameters
    # --------------------------------------------------

    def set_parameters(self, parameters):

        state_dict = self.model.state_dict()

        new_state_dict = {}

        for (key, old_value), new_value in zip(
            state_dict.items(),
            parameters
        ):

            new_state_dict[key] = torch.tensor(
                new_value,
                dtype=old_value.dtype
            )

        self.model.load_state_dict(
            new_state_dict,
            strict=True
        )


    # --------------------------------------------------
    # Local training
    # --------------------------------------------------

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        local_epochs = int(
            config.get(
                "local_epochs",
                LOCAL_EPOCHS
            )
        )

        self.model.train()

        print(
            f"\nClient {self.client_id}: "
            f"starting local training"
        )

        epoch_loss = 0.0

        for epoch in range(local_epochs):

            total_loss = 0.0
            total_samples = 0

            for X_batch, y_batch in self.loader:

                X_batch = X_batch.to(
                    DEVICE,
                    non_blocking=True
                )

                y_batch = y_batch.to(
                    DEVICE,
                    non_blocking=True
                )

                self.optimizer.zero_grad()

                outputs = self.model(
                    X_batch
                ).squeeze(1)

                loss = self.criterion(
                    outputs,
                    y_batch
                )

                loss.backward()

                self.optimizer.step()

                batch_size = X_batch.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                total_samples += batch_size

            epoch_loss = (
                total_loss / total_samples
            )

            print(
                f"Client {self.client_id} - "
                f"Epoch {epoch + 1}/{local_epochs} - "
                f"Loss: {epoch_loss:.4f}"
            )

        print(
            f"Client {self.client_id}: "
            f"local training completed"
        )

        return (
            self.get_parameters(config),
            len(self.client_indices),
            {
                "loss": float(epoch_loss)
            }
        )


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():

            for X_batch, y_batch in self.loader:

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                outputs = self.model(
                    X_batch
                ).squeeze(1)

                loss = self.criterion(
                    outputs,
                    y_batch
                )

                predictions = (
                    torch.sigmoid(outputs) >= 0.5
                ).float()

                batch_size = X_batch.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                total_correct += (
                    predictions == y_batch
                ).sum().item()

                total_samples += batch_size

        average_loss = (
            total_loss / total_samples
        )

        accuracy = (
            total_correct / total_samples
        )

        print(
            f"Client {self.client_id}: "
            f"Evaluation Loss = "
            f"{average_loss:.4f}, "
            f"Accuracy = {accuracy:.4f}"
        )

        return (
            float(average_loss),
            total_samples,
            {
                "accuracy": float(accuracy)
            }
        )


# --------------------------------------------------
# Start client
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_id",
        type=int,
        required=True
    )

    args = parser.parse_args()

    client = IDSClient(
        client_id=args.client_id
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=client
    )
