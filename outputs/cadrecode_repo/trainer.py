"""trainer.py

This module implements the Trainer class for the CAD-Recode project.
It orchestrates the end-to-end training process:
  - Batching and augmenting training samples (CAD point clouds and target CAD code tokens),
  - Forward propagation through the Model (which integrates a PointCloudProcessor and a fine-tuned LLM decoder),
  - Loss computation with Negative Log-Likelihood (NLL) loss,
  - Backpropagation and optimizer/scheduler updates,
  - Logging training progress and saving/loading training checkpoints.

Configuration parameters (learning rate, weight decay, total iterations, warmup iterations, batch size,
and logging interval) are read from the config.yaml file via a configuration dictionary.

Author: Your Name
Date: Today's Date
"""

import os
import sys
import logging
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# Ensure that the model.py and dataset_loader.py modules are in the same directory
# Trainer is designed to integrate with the Model instance and a dataset list,
# which are created by the DatasetLoader.

class CADDataset(Dataset):
    """
    A simple PyTorch Dataset class for CAD-Recode.
    Each sample is expected to be a dictionary with keys:
      - "point_cloud": a numpy.ndarray of shape (N, 3)
      - "cad_code": a string containing the CAD Python code
    """
    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        super(CADDataset, self).__init__()
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


class Trainer:
    """
    Trainer class for CAD-Recode.
    
    Attributes:
        model (torch.nn.Module): The CAD-Recode model.
        dataset (List[Dict[str, Any]]): List of training samples.
        config (dict): Configuration dictionary primarily parsed from config.yaml.
        data_loader (DataLoader): PyTorch DataLoader for batching training samples.
        optimizer (torch.optim.Optimizer): Optimizer (AdamW).
        scheduler (torch.optim.lr_scheduler._LRScheduler): Cosine Annealing scheduler with warmup.
        iteration (int): Current training iteration count.
        device (torch.device): Device to run training on.
    """
    def __init__(self, model: torch.nn.Module, dataset: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        """
        Initializes the Trainer with a model, dataset, and configuration parameters.
        
        Args:
            model (torch.nn.Module): The integrated CAD-Recode Model.
            dataset (List[Dict[str, Any]]): List of training samples.
            config (dict): Configuration dictionary.
        """
        self.model: torch.nn.Module = model
        self.config: Dict[str, Any] = config
        self.dataset_samples: List[Dict[str, Any]] = dataset
        self.device: torch.device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")
        self.model.to(self.device)

        # Retrieve training hyperparameters from config.
        training_config = config.get("training", {})
        self.learning_rate: float = training_config.get("learning_rate", 0.0002)
        self.weight_decay: float = training_config.get("weight_decay", 0.01)
        self.total_iterations: int = training_config.get("total_iterations", 100000)
        self.warmup_iterations: int = training_config.get("warmup_iterations", 1000)
        self.batch_size: int = training_config.get("batch_size", 18)
        self.log_interval: int = config.get("logging", {}).get("log_interval", 100)

        # Create a PyTorch Dataset and DataLoader.
        self.train_dataset: Dataset = CADDataset(self.dataset_samples)
        self.data_loader: DataLoader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True
        )
        self.data_loader_iter = iter(self.data_loader)

        # Set up the optimizer.
        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        # Set up the learning rate scheduler using cosine schedule with warmup.
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_iterations,
            num_training_steps=self.total_iterations
        )

        # Initialize iteration counter.
        self.iteration: int = 0

        # Use the model's tokenizer for code tokenization.
        self.tokenizer = self.model.tokenizer

        # Set up loss function: Cross Entropy Loss (for language modeling, ignore index is set to -100)
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-100)

        # Log initialization.
        logging.basicConfig(level=logging.INFO)
        logging.info("Trainer initialized with total_iterations=%d, batch_size=%d", self.total_iterations, self.batch_size)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to process a batch of samples.
        
        Each sample is expected to have:
          - "point_cloud": a numpy array of shape (N, 3)
          - "cad_code": a string containing the CAD Python code

        This function tokenizes the CAD code using the model's tokenizer and converts
        the point cloud into a torch tensor.

        Args:
            batch (List[Dict[str, Any]]): List of samples.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys:
                  "point_cloud": Tensor of shape (B, N, 3), dtype=torch.float32.
                  "code_input_ids": LongTensor of shape (B, L) with padded token IDs.
        """
        point_clouds = []
        code_texts = []
        for sample in batch:
            pc_array = sample.get("point_cloud")
            if not isinstance(pc_array, (list, tuple)) and not hasattr(pc_array, "shape"):
                raise ValueError("Each sample's point_cloud must be a numpy array or similar with attribute 'shape'.")
            point_clouds.append(torch.tensor(pc_array, dtype=torch.float32))
            code_str = sample.get("cad_code", "")
            code_texts.append(code_str)
        # Tokenize CAD code texts using the model's tokenizer.
        tokenized = self.tokenizer.batch_encode_plus(
            code_texts,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        batch_dict = {
            "point_cloud": torch.stack(point_clouds, dim=0),  # shape (B, N, 3)
            "code_input_ids": tokenized["input_ids"]  # shape (B, L)
        }
        return batch_dict

    def train(self) -> None:
        """
        Trains the CAD-Recode model using the provided dataset and configuration.
        Runs the training loop until total_iterations is reached.
        Logs training progress every log_interval iterations.
        """
        self.model.train()  # Set model to training mode.
        logging.info("Starting training for %d iterations.", self.total_iterations)
        progress_bar = tqdm(total=self.total_iterations, desc="Training", unit="iter")

        while self.iteration < self.total_iterations:
            try:
                # If the data loader is exhausted, reinitialize the iterator.
                try:
                    batch = next(self.data_loader_iter)
                except StopIteration:
                    self.data_loader_iter = iter(self.data_loader)
                    batch = next(self.data_loader_iter)

                # Move batch data to the appropriate device.
                point_cloud_batch: torch.Tensor = batch["point_cloud"].to(self.device)  # shape (B, N, 3)
                code_input_ids: torch.Tensor = batch["code_input_ids"].to(self.device)   # shape (B, L)

                # Forward pass with teacher forcing.
                outputs = self.model(point_cloud_batch, code_input_ids=code_input_ids)
                loss: torch.Tensor = outputs.loss  # Assuming the model returns a loss in training mode.

                # Backpropagation.
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.iteration += 1
                progress_bar.update(1)

                # Logging every log_interval iterations.
                if self.iteration % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    progress_bar.set_postfix({"loss": loss.item(), "lr": current_lr})
                    logging.info("Iteration %d: Loss=%.4f, LR=%.6f", self.iteration, loss.item(), current_lr)

            except Exception as e:
                logging.error("Error during training iteration %d: %s", self.iteration, str(e))
                continue

        progress_bar.close()
        logging.info("Training complete. Total iterations: %d", self.iteration)

    def save_checkpoint(self, filepath: str) -> None:
        """
        Saves a training checkpoint to the specified filepath.
        The checkpoint includes:
          - Model's state_dict.
          - Optimizer's state_dict.
          - Scheduler's state_dict.
          - Current training iteration.

        Args:
            filepath (str): Path to save the checkpoint.
        """
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "iteration": self.iteration
            }
            torch.save(checkpoint, filepath)
            logging.info("Checkpoint saved at iteration %d to '%s'", self.iteration, filepath)
        except Exception as e:
            logging.error("Failed to save checkpoint to '%s': %s", filepath, str(e))

    def load_checkpoint(self, filepath: str) -> None:
        """
        Loads a training checkpoint from the specified filepath and restores the model,
        optimizer, scheduler states, and the training iteration counter.

        Args:
            filepath (str): Path to the checkpoint file.
        """
        if not os.path.exists(filepath):
            logging.error("Checkpoint file '%s' does not exist.", filepath)
            return
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.iteration = checkpoint.get("iteration", 0)
            logging.info("Checkpoint loaded from '%s' at iteration %d", filepath, self.iteration)
        except Exception as e:
            logging.error("Failed to load checkpoint from '%s': %s", filepath, str(e))
