```python
"""
Main entry point for CAD-Recode system.
Orchestrates dataset generation, training, evaluation, and inference pipelines
for reproducing the CAD-Recode paper experiments.
"""

import argparse
import gc
import logging
import os
import random
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import AutoTokenizer

from config import Config
from data.dataset_generator import DatasetGenerator
from data.dataset_loader import CADDataset
from models.cad_recode_model import CADRecodeModel
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.cad_utils import CADUtils
from utils.point_cloud_utils import PointCloudUtils


class Main:
    """
    Main orchestrator class for CAD-Recode pipeline.
    Provides unified interface for dataset generation, training, evaluation, and inference.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize main pipeline with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load and validate configuration
        self.config = Config()
        if Path(config_path).exists():
            self.config.load_from_file(config_path)
        else:
            logging.warning(f"Configuration file not found: {config_path}, using defaults")
        
        # Setup logging
        self._setup_logging()
        
        # Setup environment
        self._setup_environment()
        
        # Initialize components (will be created as needed)
        self.dataset_generator: Optional[DatasetGenerator] = None
        self.model: Optional[CADRecodeModel] = None
        self.trainer: Optional[Trainer] = None
        self.evaluator: Optional[Evaluator] = None
        
        # Track pipeline state
        self.pipeline_stats = {
            'start_time': time.time(),
            'dataset_generation_time': 0.0,
            'training_time': 0.0,
            'evaluation_time': 0.0,
            'total_samples_processed': 0,
            'memory_peak_usage': 0.0
        }
        
        self.logger.info("CAD-Recode pipeline initialized successfully")
        self.logger.info(f"Configuration loaded from: {config_path}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Mixed precision: {getattr(self.config.training, 'mixed_precision', True)}")
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration."""
        # Create logs directory
        logs_dir = Path(self.config.paths.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_level = logging.INFO
        
        # Setup file handler
        log_file = logs_dir / f"cad_recode_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[file_handler, console_handler]
        )
        
        # Setup class logger
        self.logger = logging.getLogger(__name__)
        
        # Suppress verbose transformers logging
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        
        self.logger.info(f"Logging configured: {log_file}")
    
    def _setup_environment(self) -> None:
        """Setup environment for reproducible experiments."""
        # Set random seeds for reproducibility
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            
            # Enable deterministic operations when possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Log GPU information
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
            
            self.logger.info(f"CUDA available: {gpu_count} GPUs")
            self.logger.info(f"Current GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Validate GPU memory against requirements
            min_memory_gb = float(self.config.hardware.min_gpu_memory.replace('GB', ''))
            if gpu_memory < min_memory_gb:
                self.logger.warning(f"GPU memory ({gpu_memory:.1f} GB) below recommended "
                                  f"minimum ({min_memory_gb} GB)")
        else:
            self.logger.warning("CUDA not available, using CPU")
        
        # Create necessary directories
        self._create_directories()
        
        # Set environment variables for optimal performance
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
        
        self.logger.info(f"Environment setup complete (seed: {random_seed})")
    
    def _create_directories(self) -> None:
        """Create all necessary directories from configuration."""
        directories = [
            self.config.paths.pretrained_model_dir,
            self.config.paths.finetuned_model_dir,
            self.config.paths.raw_data_dir,
            self.config.paths.processed_data_dir,
            self.config.paths.output_dir,
            self.config.paths.logs_dir,
            self.config.training.checkpoint_dir,
            self.config.evaluation.results_dir,
            self.config.evaluation.visualizations_dir,
            self.config.dataset.train.data_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Created necessary directories")
    
    def _validate_paper_compliance(self) -> None:
        """Validate configuration compliance with paper specifications."""
        self.logger.info("Validating paper compliance...")
        
        # Critical paper parameters
        paper_specs = {
            'num_points': 256,  # Point clouds downsampled to 256 points
            'train_dataset_size': 1000000,  # One million CAD sequences
            'llm_model': 'Qwen2-1.5B',  # LLM backbone
            'loss_function': 'negative_log_likelihood'  # Training objective
        }
        
        # Validate point cloud processing
        if self.config.model.projector.num_points != paper_specs['num_points']:
            raise ValueError(f"num_points must be {paper_specs['num_points']} (from paper), "
                           f"got {self.config.model.projector.num_points}")
        
        # Validate training dataset size
        if self.config.dataset.train.size != paper_specs['train_dataset_size']:
            raise ValueError(f"Training dataset size must be {paper_specs['train_dataset_size']} "
                           f"(from paper), got {self.config.dataset.train.size}")
        
        # Validate LLM model
        if 'Qwen2-1.5B' not in self.config.model.llm_model_name:
            self.logger.warning(f"LLM model {self.config.model.llm_model_name} differs from "
                              f"paper specification (Qwen2-1.5B)")
        
        # Validate loss function
        if self.config.training.loss_function != paper_specs['loss_function']:
            raise ValueError(f"Loss function must be {paper_specs['loss_function']} (from paper), "
                           f"got {self.config.training.loss_function}")
        
        # Validate test dataset sizes (from paper)
        test_sizes = {
            'deepcad': 8046,
            'fusion360': 1725,
            'cc3d': 2973
        }
        
        for dataset_name, expected_size in test_sizes.items():
            config_size = getattr(self.config.dataset.test, dataset_name).num_samples
            if config_size != expected_size:
                self.logger.warning(f"{dataset_name} dataset size {config_size} differs "
                                  f"from paper ({expected_size})")
        
        self.logger.info("Paper compliance validation completed")
    
    def run_dataset_generation(self) -> None:
        """
        Generate procedural CAD training dataset.
        Creates 1 million CAD sketch-extrude sequences as specified in paper.
        """
        self.logger.info("Starting dataset generation...")
        start_time = time.time()
        
        try:
            # Validate configuration
            self._validate_paper_compliance()
            
            # Initialize dataset generator
            self.dataset_generator = DatasetGenerator(
                output_dir=self.config.dataset.train.data_dir,
                num_samples=self.config.dataset.train.size,
                config=self.config
            )
            
            # Check if dataset already exists
            data_dir = Path(self.config.dataset.train.data_dir)
            if data_dir.exists() and any(data_dir.iterdir()):
                response = input(f"Dataset directory {data_dir} exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info("Dataset generation cancelled")
                    return
                else:
                    shutil.rmtree(data_dir)
                    data_dir.mkdir(parents=True)
            
            # Generate dataset
            self.logger.info(f"Generating {self.config.dataset.train.size} CAD sequences...")
            self.dataset_generator.generate_dataset()
            
            # Log generation statistics
            generation_time = time.time() - start_time
            self.pipeline_stats['dataset_generation_time'] = generation_time
            
            stats = self.dataset_generator.get_generation_statistics()
            self.logger.info(f"Dataset generation completed in {generation_time:.2f}s")
            self.logger.info(f"Success rate: {stats['success_rate']:.2%}")
            self.logger.info(f"Total samples: {stats['successful_samples']}")
            self.logger.info(f"Validation failures: {stats['validation_failures']}")
            self.logger.info(f"Execution failures: {stats['execution_failures']}")
            
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {e}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Dataset generation failed: {e}")
    
    def run_training(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """
        Train CAD-Recode model following paper methodology.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume training from
        """
        self.logger.info("Starting model training...")
        start_time = time.time()
        
        try:
            # Validate configuration and dataset
            self._validate_paper_compliance()
            self._validate_training_prerequisites()
            
            # Initialize model
            self.logger.info("Initializing CAD-Recode model...")
            self.model = CADRecodeModel.from_config(self.config)
            
            # Load checkpoint if resuming
            if resume_from_checkpoint:
                self.logger.info(f"Resuming training from {resume_from_checkpoint}")
                self.model.load_model(resume_from_checkpoint)
            
            # Setup data loaders
            train_loader, val_loader = self._setup_training_data_loaders()
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config
            )
            
            # Start training
            self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs...")
            self.logger.info(f"Target training time: {self.config.hardware.training_time_estimate}")
            
            self.trainer.train(self.config.training.num_epochs)
            
            # Save final model
            final_model_path = Path(self.config.paths.finetuned_model_dir) / "final_model"
            self.model.save_model(str(final_model_path))
            
            # Log training statistics
            training_time = time.time() - start_time
            self.pipeline_stats['training_time'] = training_time
            
            self.logger.info(f"Training completed in {training_time:.2f}s ({training_time/3600:.2f}h)")
            self.logger.info(f"Final model saved to: {final_model_path}")
            
            # Validate training time against paper estimate
            paper_time_hours = 12  # From paper: ~12 hours on H100
            actual_time_hours = training_time / 3600
            if actual_time_hours > paper_time_hours * 2:  # Allow 2x variance
                self.logger.warning(f"Training time ({actual_time_hours:.2f}h) significantly "
                                  f"exceeds paper estimate ({paper_time_hours}h)")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Training failed: {e}")
    
    def _validate_training_prerequisites(self) -> None:
        """Validate prerequisites for training."""
        # Check if training dataset exists
        data_dir = Path(self.config.dataset.train.data_dir)
        if not data_dir.exists() or not any(data_dir.iterdir()):
            raise FileNotFoundError(f"Training dataset not found at {data_dir}. "
                                  f"Run dataset generation first.")
        
        # Check dataset size
        point_clouds_dir = data_dir / "point_clouds"
        cad_codes_dir = data_dir / "cad_codes"
        
        if not point_clouds_dir.exists() or not cad_codes_dir.exists():
            raise FileNotFoundError("Dataset directories incomplete. "
                                   "Re-run dataset generation.")
        
        # Count samples
        pc_files = list(point_clouds_dir.glob("*.pt"))
        code_files = list(cad_codes_dir.glob("*.py"))
        
        min_samples = min(len(pc_files), len(code_files))
        if min_samples < 1000:  # Minimum viable dataset size
            raise ValueError(f"Insufficient training samples: {min_samples}. "
                           f"Expected at least 1000.")
        
        self.logger.info(f"Training dataset validated: {min_samples} samples")
    
    def _setup_training_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Setup training and validation data loaders."""
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.llm_model_name,
            trust_remote_code=True
        )