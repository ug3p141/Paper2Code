```python
"""
Experimental orchestration for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the central experimental coordinator that manages all experiments
described in the paper: training, evaluation, ablation studies, and CAD-QA experiments.
It ensures reproducible execution of all experimental workflows with proper result
collection and comparison against baseline methods.
"""

import os
import sys
import json
import time
import logging
import warnings
import traceback
import gc
import shutil
import signal
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
import threading
import multiprocessing as mp

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Wandb not available. Logging will be limited.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    raise ImportError("Transformers library is required for experiments")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI library not available. CAD-QA experiments will be limited.")

# Import project modules
from config import Config
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from data.dataset_generator import DatasetGenerator
from data.dataset_loader import DatasetLoader
from models.cad_recode_model import CADRecodeModel
from evaluation.metrics import Metrics
from utils.cad_validation import CADValidator


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Data structure for storing experiment results."""
    experiment_type: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    results: Dict[str, Any]
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class TrainingResult:
    """Training experiment result structure."""
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    training_time: float
    total_iterations: int
    model_checkpoint_path: str
    training_metrics_history: List[Dict[str, Any]]
    convergence_iteration: Optional[int] = None


@dataclass
class EvaluationResult:
    """Evaluation experiment result structure."""
    dataset_results: Dict[str, Dict[str, float]]
    comparison_with_baselines: Dict[str, Dict[str, float]]
    qualitative_examples: List[Dict[str, Any]]
    evaluation_time: float
    total_samples_evaluated: int


@dataclass
class AblationResult:
    """Ablation study result structure."""
    ablation_type: str
    configurations: List[Dict[str, Any]]
    results: List[Dict[str, float]]
    best_configuration: Dict[str, Any]
    performance_trends: Dict[str, List[float]]


class ExperimentRunner:
    """
    Central experimental orchestrator for CAD-Recode reproduction.
    
    Manages all experimental workflows described in the paper:
    - Training experiments with procedural dataset
    - Evaluation on DeepCAD, Fusion360, and CC3D datasets
    - Ablation studies on data, architecture, and sampling
    - CAD-QA experiments with GPT-4o integration
    - Interactive editing demonstrations
    """
    
    def __init__(self, config: Config):
        """
        Initialize experiment runner with configuration.
        
        Args:
            config: Configuration object from config.py
        """
        # Store configuration
        self.config = config
        
        # Extract key configuration sections
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        self.data_config = config.get_data_config()
        self.eval_config = config.get_evaluation_config()
        self.system_config = config.get_system_config()
        self.paths_config = config.get_paths_config()
        self.logging_config = config.get_logging_config()
        
        # Set up device management
        self.device = self.system_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize paths
        self.base_dir = Path(self.paths_config.get('data_dir', './data'))
        self.checkpoints_dir = Path(self.paths_config.get('checkpoints_dir', './checkpoints'))
        self.results_dir = Path(self.paths_config.get('results_dir', './results'))
        self.logs_dir = Path(self.paths_config.get('logs_dir', './logs'))
        
        # Create directories
        for directory in [self.base_dir, self.checkpoints_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracking
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.current_experiment = None
        self.start_time = None
        
        # Initialize shared components
        self.tokenizer = None
        self.model = None
        self.dataset_generator = None
        self.dataset_loader = None
        
        # Set random seeds for reproducibility
        self._set_random_seeds(42)
        
        # Initialize logging
        self._setup_experiment_logging()
        
        # Initialize baseline results for comparison
        self.baseline_results = self._load_baseline_results()
        
        logger.info(f"ExperimentRunner initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Results directory: {self.results_dir}")
        logger.info(f"  Checkpoints directory: {self.checkpoints_dir}")
        logger.info(f"  Available experiments: training, evaluation, ablation, cad_qa, editing")
    
    def _set_random_seeds(self, seed: int = 42) -> None:
        """Set random seeds for reproducibility across all experiments."""
        import random
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Random seeds set to {seed} for reproducibility")
    
    def _setup_experiment_logging(self) -> None:
        """Set up logging configuration for experiments."""
        # Create experiment-specific log file
        log_file = self.logs_dir / f"experiments_{int(time.time())}.log"
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Initialize wandb if available and configured
        if WANDB_AVAILABLE and self.logging_config.get('wandb_project'):
            try:
                wandb.init(
                    project=self.logging_config['wandb_project'],
                    config=asdict(self.config) if hasattr(self.config, '__dict__') else {},
                    dir=str(self.logs_dir),
                    name=f"cad-recode-experiments-{int(time.time())}"
                )
                self.use_wandb = True
                logger.info("Wandb logging initialized for experiments")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def _load_baseline_results(self) -> Dict[str, Dict[str, float]]:
        """
        Load baseline method results for comparison.
        
        Returns:
            Dictionary with baseline results from paper Table 1 and Table 2
        """
        # Baseline results from Table 1 (DeepCAD and Fusion360)
        baseline_results = {
            'DeepCAD': {
                'DeepCAD': {'mean_cd': 42.5, 'med_cd': 9.64, 'iou': 46.7, 'ir': 7.1},
                'PrismCAD': {'mean_cd': float('nan'), 'med_cd': 4.28, 'iou': 72.1, 'ir': 16.2},
                'Point2Cyl': {'mean_cd': float('nan'), 'med_cd': 4.27, 'iou': 73.8, 'ir': 3.9},
                'HNC-CAD': {'mean_cd': float('nan'), 'med_cd': 8.64, 'iou': 65.3, 'ir': 5.6},
                'MultiCAD': {'mean_cd': float('nan'), 'med_cd': 8.09, 'iou': float('nan'), 'ir': 11.5},
                'TransCAD': {'mean_cd': 32.3, 'med_cd': 4.51, 'iou': 65.5, 'ir': 1.1},
                'CAD-Diffuser': {'mean_cd': float('nan'), 'med_cd': 3.02, 'iou': 74.3, 'ir': 1.5},
                'CAD-SIGNet': {'mean_cd': 3.43, 'med_cd': 0.28, 'iou': 77.6, 'ir': 0.9}
            },
            'Fusion360': {
                'DeepCAD': {'mean_cd': 330, 'med_cd': 89.2, 'iou': 39.9, 'ir': 25.2},
                'PrismCAD': {'mean_cd': float('nan'), 'med_cd': 4.75, 'iou': 65.3, 'ir': 18.0},
                'Point2Cyl': {'mean_cd': float('nan'), 'med_cd': 4.18, 'iou': 67.5, 'ir': 3.2},
                'HNC-CAD': {'mean_cd': float('nan'), 'med_cd': 36.8, 'iou': 63.5, 'ir': 7.3},
                'MultiCAD': {'mean_cd': float('nan'), 'med_cd': 42.2, 'iou': float('nan'), 'ir': 16.5},
                'TransCAD': {'mean_cd': 78.6, 'med_cd': 33.4, 'iou': 60.2, 'ir': 2.4},
                'CAD-Diffuser': {'mean_cd': float('nan'), 'med_cd': 3.85, 'iou': 63.2, 'ir': 1.7},
                'CAD-SIGNet': {'mean_cd': 7.37, 'med_cd': 0.48, 'iou': 65.6, 'ir': 1.6}
            },
            'CC3D': {
                'DeepCAD': {'mean_cd': float('nan'), 'med_cd': 263, 'iou': float('nan'), 'ir': 12.7},
                'CAD-SIGNet': {'mean_cd': 14.82, 'med_cd': 2.90, 'iou': 42.6, 'ir': 2.5}
            }
        }
        
        return baseline_results
    
    def run_training_experiment(self) -> ExperimentResult:
        """
        Execute main training experiment from Section 4.3 of the paper.
        
        Trains CAD-Recode on 1M procedurally generated CAD sequences using
        end-to-end strategy with point cloud projector from scratch and
        LLM fine-tuning.
        
        Returns:
            ExperimentResult containing training outcomes and metrics
        """
        logger.info("Starting training experiment...")
        start_time = time.time()
        
        try:
            # Setup experiment environment
            self._setup_experiment_environment('training')
            
            # Step 1: Initialize or load procedural dataset
            logger.info("Initializing procedural dataset generation...")
            self.dataset_generator = DatasetGenerator(asdict(self.config) if hasattr(self.config, '__dict__') else {})
            
            # Generate dataset if not already available
            dataset_path = self.base_dir / 'procedural_dataset' / 'dataset.pkl'
            if not dataset_path.exists():
                logger.info("Generating 1M procedural CAD sequences...")
                dataset = self.dataset_generator.generate_dataset(
                    size=self.data_config['procedural_dataset']['size']
                )
                
                # Save generated dataset
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                with open(dataset_path, 'wb') as f:
                    import pickle
                    pickle.dump(dataset, f)
                logger.info(f"Dataset saved to {dataset_path}")
            else:
                logger.info(f"Loading existing dataset from {dataset_path}")
            
            # Step 2: Initialize tokenizer
            logger.info("Initializing tokenizer...")
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers library required for training")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['llm_model_name'],
                trust_remote_code=True
            )
            
            # Add special tokens
            special_tokens = {
                'additional_special_tokens': [
                    self.model_config['start_token'],
                    self.model_config['end_token']
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)
            
            # Step 3: Initialize dataset loader
            logger.info("Initializing dataset loader...")
            self.dataset_loader = DatasetLoader(
                config=asdict(self.config) if hasattr(self.config, '__dict__') else {},
                tokenizer=self.tokenizer
            )
            
            # Load training and validation data
            train_dataset = self.dataset_loader.load_procedural_data(split='train')
            val_dataset = self.dataset_loader.load_procedural_data(split='val')
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True,
                num_workers=self.system_config.get('num_workers', 4),
                pin_memory=self.system_config.get('pin_memory', True),
                drop_last=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                num_workers=self.system_config.get('num_workers', 4),
                pin_memory=self.system_config.get('pin_memory', True),
                drop_last=False
            )
            
            logger.info(f"Data loaders created: train={len(train_loader)}, val={len(val_loader)}")
            
            # Step 4: Initialize model
            logger.info("Initializing CAD-Recode model...")
            self