```python
"""
Main entry point for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module serves as the central orchestrator for the complete CAD-Recode experimental
pipeline, managing configuration loading, logging setup, and experiment execution.
It provides a clean command-line interface for reproducing all paper results and
conducting additional experiments.
"""

import argparse
import logging
import os
import sys
import time
import traceback
import signal
import gc
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

import torch
import numpy as np
import psutil

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

# Import project modules
from config import Config
from experiments.run_experiments import ExperimentRunner


class Main:
    """
    Central orchestrator for CAD-Recode experimental pipeline.
    
    Manages the complete workflow from configuration loading through experiment
    execution, providing comprehensive logging, error handling, and result
    collection for reproducible scientific experiments.
    """
    
    def __init__(self):
        """Initialize main orchestrator with default settings."""
        # Core attributes
        self.config: Optional[Config] = None
        self.logger: Optional[logging.Logger] = None
        self.experiment_runner: Optional[ExperimentRunner] = None
        self.start_time: Optional[float] = None
        
        # Experiment state tracking
        self.experiment_results: Dict[str, Any] = {}
        self.system_info: Dict[str, Any] = {}
        self.performance_stats: Dict[str, Any] = {}
        
        # Signal handling for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize argument parser
        self.parser = self._create_argument_parser()
        
        # Wandb integration state
        self.use_wandb = False
        self.wandb_run = None
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Shutting down gracefully...")
            self._cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """
        Create command-line argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description="CAD-Recode: Reverse Engineering CAD Code from Point Clouds",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all experiments with default config
  python main.py --experiment all
  
  # Run only training experiment
  python main.py --experiment training
  
  # Evaluate on specific dataset
  python main.py --experiment evaluation --dataset deepcad
  
  # Run ablation studies
  python main.py --experiment ablation
  
  # Debug mode with verbose logging
  python main.py --debug --experiment training
  
  # Use custom configuration
  python main.py --config custom_config.yaml --experiment all
            """
        )
        
        # Configuration arguments
        parser.add_argument(
            '--config', 
            type=str, 
            default='config.yaml',
            help='Path to configuration file (default: config.yaml)'
        )
        
        # Experiment selection
        parser.add_argument(
            '--experiment',
            type=str,
            choices=['training', 'evaluation', 'ablation', 'cad_qa', 'editing', 'all'],
            default='all',
            help='Type of experiment to run (default: all)'
        )
        
        # Dataset specification for evaluation
        parser.add_argument(
            '--dataset',
            type=str,
            choices=['procedural', 'deepcad', 'fusion360', 'cc3d'],
            help='Specific dataset for evaluation experiments'
        )
        
        # Model checkpoint
        parser.add_argument(
            '--model_path',
            type=str,
            help='Path to pre-trained model checkpoint for evaluation'
        )
        
        # Output directory override
        parser.add_argument(
            '--output_dir',
            type=str,
            help='Override output directory from config'
        )
        
        # Debug and logging options
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode with verbose logging'
        )
        
        parser.add_argument(
            '--log_level',
            type=str,
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set logging level (default: INFO)'
        )
        
        # Reproducibility
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='Random seed for reproducibility (default: 42)'
        )
        
        # Resource management
        parser.add_argument(
            '--device',
            type=str,
            choices=['auto', 'cuda', 'cpu'],
            default='auto',
            help='Device to use for computation (default: auto)'
        )
        
        parser.add_argument(
            '--batch_size',
            type=int,
            help='Override batch size from config'
        )
        
        parser.add_argument(
            '--num_workers',
            type=int,
            help='Override number of data loading workers'
        )
        
        # Experiment-specific options
        parser.add_argument(
            '--skip_training',
            action='store_true',
            help='Skip training and use existing checkpoint for evaluation'
        )
        
        parser.add_argument(
            '--skip_dataset_generation',
            action='store_true',
            help='Skip dataset generation if dataset already exists'
        )
        
        parser.add_argument(
            '--quick_test',
            action='store_true',
            help='Run experiments with reduced dataset size for quick testing'
        )
        
        # Wandb options
        parser.add_argument(
            '--wandb_project',
            type=str,
            help='Wandb project name override'
        )
        
        parser.add_argument(
            '--wandb_name',
            type=str,
            help='Wandb run name'
        )
        
        parser.add_argument(
            '--no_wandb',
            action='store_true',
            help='Disable wandb logging'
        )
        
        return parser
    
    def setup_logging(self) -> None:
        """
        Setup comprehensive logging system based on configuration.
        
        Configures multi-level logging with file and console handlers,
        integrates with wandb for experiment tracking, and records
        complete environment information for reproducibility.
        """
        if self.config is None:
            raise RuntimeError("Configuration must be loaded before setting up logging")
        
        # Get logging configuration
        logging_config = self.config.get_logging_config()
        paths_config = self.config.get_paths_config()
        
        # Create logs directory
        logs_dir = Path(paths_config.get('logs_dir', './logs'))
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler with rotation
        log_file = logs_dir / f"cad_recode_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = getattr(logging, self.args.log_level.upper())
        console_handler.setLevel(console_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # Get logger for this module
        self.logger = logging.getLogger(__name__)
        
        # Log system information
        self._log_system_info()
        
        # Initialize wandb if enabled
        self._setup_wandb_logging()
        
        self.logger.info("Logging system initialized successfully")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Console log level: {self.args.log_level}")
    
    def _log_system_info(self) -> None:
        """Log comprehensive system and environment information."""
        if self.logger is None:
            return
        
        # Collect system information
        self.system_info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            },
            'hardware': {
                'cpu_count': os.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'disk_space_gb': round(psutil.disk_usage('.').total / (1024**3), 2),
            },
            'pytorch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': gpu_props.name,
                    'memory_gb': round(gpu_props.total_memory / (1024**3), 2),
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                })
            self.system_info['gpus'] = gpu_info
        
        # Add Git information if available
        if GIT_AVAILABLE:
            try:
                repo = git.Repo(search_parent_directories=True)
                self.system_info['git'] = {
                    'commit_hash': repo.head.object.hexsha,
                    'branch': repo.active_branch.name,
                    'is_dirty': repo.is_dirty(),
                    'remote_url': repo.remotes.origin.url if repo.remotes else None
                }
            except Exception as e:
                self.logger.warning(f"Could not get Git information: {e}")
        
        # Log system information
        self.logger.info("=== SYSTEM INFORMATION ===")
        self.logger.info(f"Platform: {self.system_info['platform']['system']} {self.system_info['platform']['release']}")
        self.logger.info(f"Python: {self.system_info['platform']['python_version']}")
        self.logger.info(f"PyTorch: {self.system_info['pytorch']['version']}")
        self.logger.info(f"CPU cores: {self.system_info['hardware']['cpu_count']}")
        self.logger.info(f"Memory: {self.system_info['hardware']['memory_gb']} GB")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA: {self.system_info['pytorch']['cuda_version']}")
            for i, gpu in enumerate(self.system_info.get('gpus', [])):
                self.logger.info(f"GPU {i}: {gpu['name']} ({gpu['memory_gb']} GB)")
        else:
            self.logger.info("CUDA: Not available")
        
        if 'git' in self.system_info:
            git_info = self.system_info['git']
            self.logger.info(f"Git: {git_info['branch']}@{git_info['commit_hash'][:8]}")
            if git_info['is_dirty']:
                self.logger.warning("Git repository has uncommitted changes")
        
        self.logger.info("=== END SYSTEM INFORMATION ===")
    
    def _setup_wandb_logging(self) -> None:
        """Setup Weights & Biases experiment tracking."""
        if not WANDB_AVAILABLE or self.args.no_wandb:
            self.logger.info("Wandb logging disabled")
            return
        
        try:
            # Get wandb configuration
            logging_config = self.config.get_logging_config()
            wandb_project = self.args.wandb_project or logging_config.get('wandb_project', 'cad-recode')
            
            if not wandb_project:
                self.logger.info("No wandb project specified, skipping wandb initialization")
                return
            
            # Prepare wandb config
            wandb_config = {
                'experiment_type': self.args.experiment,
                'system_info': self.system_info,
                'config': self.config._config if hasattr(self.config, '_config') else {},
                'command_line_args': vars(self.args)
            }
            
            # Initialize wandb
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=self.args.wandb_name or f"cad-recode-{self.args.experiment}-{int(time.time())}",
                config=wandb_config,
                dir=str(Path(self.config.get_paths_config().get('logs_dir', './logs'))),
                tags=[self.args.experiment, f"seed-{self.args.seed}"],
                notes=f"CAD-Recode experiment: {self.args.experiment}"
            )
            
            self.use_wandb = True
            self.logger.info(f"Wandb initialized: {wandb_project}/{self.wandb_run.name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def _load_and_validate_config(self) -> None:
        """
        Load configuration file and validate all parameters.
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = Path(self.args.config)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        self.config = Config.load_from_file(str(config_path))
        
        # Apply command-line overrides
        self._apply_config_overrides()
        
        # Validate configuration
        try:
            self.config.validate_parameters()
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def _apply_config_overrides(self) -> None:
        """Apply command-line argument overrides to configuration."""
        if self.config is None:
            return
        
        # Device override
        if self.args.device != 'auto':
            self.config.update_parameter('system', 'device