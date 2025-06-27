```python
"""
Trainer module for CAD-Recode system.
Manages the complete training process with negative log-likelihood loss for CAD code generation.
"""

import gc
import logging
import math
import os
import shutil
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm

from config import Config
from models.cad_recode_model import CADRecodeModel


class Trainer:
    """
    Trainer class for CAD-Recode model training.
    Implements complete training pipeline with negative log-likelihood loss, mixed precision,
    gradient accumulation, and comprehensive logging.
    """
    
    def __init__(self, 
                 model: CADRecodeModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Optional[Config] = None):
        """
        Initialize trainer with model and data loaders.
        
        Args:
            model: CADRecodeModel instance to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object with training parameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config if config is not None else Config()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training configuration from config
        self.learning_rate = self.config.training.learning_rate
        self.batch_size = self.config.training.batch_size
        self.gradient_accumulation_steps = self.config.training.gradient_accumulation_steps
        self.num_epochs = self.config.training.num_epochs
        self.warmup_steps = self.config.training.warmup_steps
        self.weight_decay = self.config.training.weight_decay
        self.mixed_precision = self.config.training.mixed_precision
        self.max_grad_norm = getattr(self.config.training, 'max_grad_norm', 1.0)
        
        # Logging and checkpointing
        self.log_every_n_steps = self.config.training.log_every_n_steps
        self.eval_every_n_epochs = self.config.training.eval_every_n_epochs
        self.save_every_n_epochs = self.config.training.save_every_n_epochs
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device(self.config.training.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, using CPU")
            self.device = torch.device('cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.total_training_steps = 0
        
        # Performance tracking
        self.training_stats = {
            'epoch_losses': [],
            'validation_losses': [],
            'learning_rates': [],
            'training_times': [],
            'memory_usage': [],
            'gradient_norms': []
        }
        
        # Calculate total training steps for scheduler
        self.total_training_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        
        self.logger.info(f"Initialized Trainer: device={self.device}, "
                        f"effective_batch_size={self.batch_size * self.gradient_accumulation_steps}, "
                        f"total_steps={self.total_training_steps}")
    
    def _setup_optimizer(self) -> None:
        """Initialize optimizer with proper parameter grouping."""
        # Separate parameters for different learning rates if needed
        param_groups = []
        
        # Point cloud projector parameters
        projector_params = list(self.model.projector.parameters())
        if projector_params:
            param_groups.append({
                'params': projector_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'name': 'projector'
            })
        
        # LLM parameters (might want different learning rate)
        llm_params = []
        if hasattr(self.model, 'llm'):
            llm_params.extend(list(self.model.llm.parameters()))
        if hasattr(self.model, 'lm_head'):
            llm_params.extend(list(self.model.lm_head.parameters()))
        
        if llm_params:
            param_groups.append({
                'params': llm_params,
                'lr': self.learning_rate * 0.1,  # Lower learning rate for pre-trained LLM
                'weight_decay': self.weight_decay,
                'name': 'llm'
            })
        
        # Fallback: all parameters with same settings
        if not param_groups:
            param_groups = [{
                'params': self.model.parameters(),
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'name': 'all'
            }]
        
        self.optimizer = AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.logger.info(f"Optimizer initialized with {len(param_groups)} parameter groups")
    
    def _setup_scheduler(self) -> None:
        """Initialize learning rate scheduler with warmup."""
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.total_training_steps - self.warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        self.logger.info(f"Scheduler initialized: warmup_steps={self.warmup_steps}, "
                        f"total_steps={self.total_training_steps}")
    
    def _setup_mixed_precision(self) -> None:
        """Initialize mixed precision training components."""
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            self.use_amp = False
            if self.mixed_precision:
                self.logger.warning("Mixed precision requested but CUDA not available")
    
    def _setup_logging(self) -> None:
        """Initialize logging and monitoring."""
        # TensorBoard logging
        log_dir = Path(self.config.paths.logs_dir) / f"training_{int(time.time())}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        self.logger.info(f"TensorBoard logging to {log_dir}")
    
    def train(self, num_epochs: int) -> None:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        try:
            # Training loop
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                epoch_start_time = time.time()
                train_loss = self.train_epoch()
                epoch_time = time.time() - epoch_start_time
                
                # Update statistics
                self.training_stats['epoch_losses'].append(train_loss)
                self.training_stats['training_times'].append(epoch_time)
                
                # Log epoch results
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs} completed: "
                                f"train_loss={train_loss:.6f}, time={epoch_time:.2f}s")
                
                # Validation phase
                if (epoch + 1) % self.eval_every_n_epochs == 0:
                    val_loss = self.validate()
                    self.training_stats['validation_losses'].append(val_loss)
                    
                    # Check for best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, val_loss, is_best=True)
                        self.logger.info(f"New best validation loss: {val_loss:.6f}")
                
                # Save checkpoint
                if (epoch + 1) % self.save_every_n_epochs == 0:
                    self.save_checkpoint(epoch, train_loss)
                
                # Log to TensorBoard
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Train/EpochTime', epoch_time, epoch)
                if self.training_stats['validation_losses']:
                    self.writer.add_scalar('Validation/Loss', 
                                         self.training_stats['validation_losses'][-1], epoch)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            self.logger.info("Training completed successfully")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_checkpoint(self.current_epoch, train_loss, is_interrupted=True)
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            self.logger.error(traceback.format_exc())
            self.save_checkpoint(self.current_epoch, train_loss, is_error=True)
            raise
        finally:
            # Cleanup
            if hasattr(self, 'writer'):
                self.writer.close()
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(pbar):
            try:
                # Unpack batch data
                if len(batch_data) == 3:
                    point_clouds, input_ids, attention_mask = batch_data
                else:
                    raise ValueError(f"Expected 3 elements in batch, got {len(batch_data)}")
                
                # Move to device
                point_clouds = point_clouds.to(self.device, non_blocking=True)
                input_ids = input_ids.to(self.device, non_blocking=True)
                
                # Validate batch
                if point_clouds.size(0) == 0:
                    self.logger.warning(f"Empty batch at step {batch_idx}, skipping")
                    continue
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # CADRecodeModel expects (point_clouds, target_codes)
                    # For training, target_codes should be input_ids
                    loss = self.model(point_clouds, input_ids)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Check for invalid loss
                if not torch.isfinite(loss):
                    self.logger.warning(f"Invalid loss detected at step {batch_idx}: {loss}")
                    continue
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate loss
                accumulated_loss += loss.item()
                
                # Optimization step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    # Scheduler step
                    self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.log_every_n_steps == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        self.training_stats['learning_rates'].append(current_lr)
                        self.training_stats['gradient_norms'].append(grad_norm.item())
                        
                        # TensorBoard logging
                        self.writer.add_scalar('Train/StepLoss', accumulated_loss, self.global_step)
                        self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
                        self.writer.add_scalar('Train/GradientNorm', grad_norm.item(), self.global_step)
                        
                        # Memory usage
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                            self.training_stats['memory_usage'].append(memory_used)
                            self.writer.add_scalar('System/MemoryUsage', memory_used, self.global_step)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{accumulated_loss:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                        'step': self.global_step
                    })
                    
                    # Accumulate for epoch average
                    total_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                self.logger.error(traceback.format_exc())
                
                # Skip this batch and continue
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0: