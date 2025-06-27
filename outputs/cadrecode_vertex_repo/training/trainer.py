```python
"""
Training module for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the end-to-end training strategy from Section 4.3 of the paper,
training the point cloud projector from scratch while fine-tuning the pre-trained LLM
using Negative Log-Likelihood loss on procedurally generated CAD code sequences.

Training Configuration (from paper):
- AdamW optimizer: lr=0.0002, weight_decay=0.01
- Cosine learning rate scheduler with 1k warmup steps
- 100k iterations total, batch size 18
- Single stage end-to-end training
"""

import os
import sys
import json
import time
import logging
import warnings
import signal
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

try:
    from transformers import get_cosine_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Using fallback scheduler.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Wandb not available. Logging will be limited.")

# Import project modules
from models.cad_recode_model import CADRecodeModel


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics tracking structure."""
    iteration: int = 0
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    gpu_memory_mb: float = 0.0
    batch_time: float = 0.0
    data_time: float = 0.0
    samples_per_second: float = 0.0


@dataclass
class ValidationResults:
    """Validation results structure."""
    avg_loss: float
    perplexity: float
    generated_samples: List[str]
    syntax_valid_ratio: float
    geometric_valid_ratio: float
    avg_code_length: float
    total_time: float


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
        return self.stopped


class GradientClipping:
    """Gradient clipping utility."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> float:
        if self.max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.max_norm, 
                norm_type=self.norm_type
            ).item()
        else:
            # Just compute norm without clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(self.norm_type)
                    total_norm += param_norm.item() ** self.norm_type
            return total_norm ** (1.0 / self.norm_type)


class Trainer:
    """
    Training orchestrator for CAD-Recode model.
    
    Implements the end-to-end training strategy from Section 4.3:
    - Point cloud projector trained from scratch
    - Pre-trained LLM fine-tuned for CAD code generation
    - NLL loss with special token handling
    - AdamW optimizer with cosine scheduler
    """
    
    def __init__(
        self,
        model: CADRecodeModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ):
        """
        Initialize trainer with model, data loaders, and configuration.
        
        Args:
            model: CADRecodeModel instance
            train_loader: Training data loader
            val_loader: Validation data loader  
            config: Configuration dictionary from config.yaml
        """
        # Store core components
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Extract training configuration
        training_config = config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 0.0002)
        self.weight_decay = training_config.get('weight_decay', 0.01)
        self.num_iterations = training_config.get('num_iterations', 100000)
        self.warmup_steps = training_config.get('warmup_steps', 1000)
        self.batch_size = training_config.get('batch_size', 18)
        self.gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)
        
        # Extract system configuration
        system_config = config.get('system', {})
        self.device = system_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = system_config.get('mixed_precision', False)
        
        # Extract logging configuration
        logging_config = config.get('logging', {})
        self.log_interval = logging_config.get('log_interval', 100)
        self.save_interval = logging_config.get('save_interval', 5000)
        self.wandb_project = logging_config.get('wandb_project', 'cad-recode')
        
        # Extract paths configuration
        paths_config = config.get('paths', {})
        self.checkpoints_dir = Path(paths_config.get('checkpoints_dir', './checkpoints'))
        self.logs_dir = Path(paths_config.get('logs_dir', './logs'))
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self._initialize_optimizer()
        
        # Initialize scheduler
        self._initialize_scheduler()
        
        # Initialize training utilities
        self.gradient_clipper = GradientClipping(max_norm=self.gradient_clip_norm)
        self.early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
        
        # Initialize mixed precision if enabled
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.current_iteration = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None
        
        # Metrics tracking
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.recent_losses = deque(maxlen=100)  # For smoothed loss tracking
        
        # Initialize logging
        self._initialize_logging()
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Weight decay: {self.weight_decay}")
        logger.info(f"  Total iterations: {self.num_iterations:,}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {self.mixed_precision}")
    
    def _initialize_optimizer(self) -> None:
        """Initialize AdamW optimizer with paper-specified parameters."""
        # Separate parameter groups for different learning strategies
        # Point cloud projector: trained from scratch
        # LLM: fine-tuned from pre-trained weights
        
        projector_params = list(self.model.point_projector.parameters())
        llm_params = list(self.model.llm.parameters())
        
        # Add embedding projection parameters if they exist
        if hasattr(self.model, 'embedding_projection') and self.model.embedding_projection is not None:
            projector_params.extend(list(self.model.embedding_projection.parameters()))
        
        # Create parameter groups
        param_groups = [
            {
                'params': projector_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'name': 'point_projector'
            },
            {
                'params': llm_params,
                'lr': self.learning_rate,  # Same LR as specified in paper
                'weight_decay': self.weight_decay,
                'name': 'llm'
            }
        ]
        
        # Initialize AdamW with HuggingFace defaults
        self.optimizer = AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),  # HuggingFace defaults
            eps=1e-8,
            amsgrad=False
        )
        
        logger.info(f"Optimizer initialized:")
        logger.info(f"  Point projector params: {len(projector_params):,}")
        logger.info(f"  LLM params: {len(llm_params):,}")
        logger.info(f"  Total params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _initialize_scheduler(self) -> None:
        """Initialize cosine learning rate scheduler with warmup."""
        if TRANSFORMERS_AVAILABLE:
            # Use HuggingFace scheduler for consistency
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.num_iterations
            )
        else:
            # Fallback to PyTorch scheduler
            self.scheduler = CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.num_iterations - self.warmup_steps,
                eta_min=self.learning_rate * 0.01  # 1% of initial LR
            )
        
        logger.info(f"Scheduler initialized: warmup={self.warmup_steps}, total={self.num_iterations}")
    
    def _initialize_logging(self) -> None:
        """Initialize wandb logging if available."""
        if WANDB_AVAILABLE:
            try:
                # Initialize wandb
                wandb.init(
                    project=self.wandb_project,
                    config=self.config,
                    name=f"cad-recode-{int(time.time())}",
                    dir=str(self.logs_dir)
                )
                
                # Log model architecture
                wandb.watch(self.model, log='all', log_freq=self.log_interval)
                
                self.use_wandb = True
                logger.info("Wandb logging initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
            logger.info("Wandb not available, using local logging only")
    
    def train_epoch(self) -> float:
        """
        Execute one training epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        
        epoch_losses = []
        epoch_start_time = time.time()
        data_time_total = 0.0
        batch_time_total = 0.0
        
        # Progress bar for epoch
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            disable=not logger.isEnabledFor(logging.INFO)
        )
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            try:
                # Move batch to device
                data_start_time = time.time()
                point_cloud = batch['point_cloud'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                # Get labels for training
                if 'labels' in batch:
                    labels = batch['labels'].to(self.device, non_blocking=True)
                else:
                    # Create labels by shifting input_ids
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]  # Shift left
                    labels[:, -1] = -100  # Ignore last token in loss
                
                data_time = time.time() - data_start_time
                data_time_total += data_time
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if self.mixed_precision and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            point_cloud=point_cloud,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True
                        )
                        loss = outputs['loss']
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = self.gradient_clipper(self.model)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    # Standard forward pass
                    outputs = self.model(
                        point_cloud=point_cloud,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True
                    )
                    loss = outputs['loss']
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    grad_norm = self.gradient_clipper(self.model)
                    
                    # Optimizer step
                    self.optimizer.step()
                
                # Update learning rate scheduler
                self.scheduler.step()
                
                # Track metrics
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.learning_rate
                batch_time = time.time() - batch_start_time
                batch_time_total += batch_time
                
                # Store loss