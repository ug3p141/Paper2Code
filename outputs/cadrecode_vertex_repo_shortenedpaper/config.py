```python
"""
Configuration management for CAD-Recode system.
Handles loading, validation, and access to all system parameters.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import torch


class NestedConfig:
    """Helper class to enable nested attribute access for configuration dictionaries."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize nested configuration with dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, NestedConfig(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert nested configuration back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, NestedConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        """String representation of nested configuration."""
        return f"NestedConfig({self.to_dict()})"


class Config:
    """
    Configuration management class for CAD-Recode system.
    Provides structured access to all system parameters with validation.
    """
    
    def __init__(self):
        """Initialize configuration with default values."""
        self._setup_defaults()
        self._device = self._detect_device()
    
    def _setup_defaults(self) -> None:
        """Set up default configuration values based on paper specifications."""
        # Model configuration defaults
        self.model = NestedConfig({
            'llm_model_name': 'Qwen/Qwen2-1.5B',
            'projector': {
                'num_points': 256,  # From paper: point clouds downsampled to 256 points
                'hidden_dim': 768,  # Standard transformer dimension
                'fourier_freqs': 10  # Number of Fourier frequency bands
            },
            'generation': {
                'max_length': 512,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9
            }
        })
        
        # Training configuration defaults
        self.training = NestedConfig({
            'learning_rate': 5e-5,  # Standard LLM fine-tuning rate
            'batch_size': 8,  # Memory-efficient for LLM training
            'gradient_accumulation_steps': 4,  # Effective batch size of 32
            'num_epochs': 10,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'loss_function': 'negative_log_likelihood',  # From paper
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mixed_precision': True,
            'save_every_n_epochs': 1,
            'checkpoint_dir': './checkpoints',
            'log_every_n_steps': 100,
            'eval_every_n_epochs': 1
        })
        
        # Dataset configuration defaults
        self.dataset = NestedConfig({
            'train': {
                'size': 1000000,  # One million CAD sequences from paper
                'data_dir': './data/generated'
            },
            'test': {
                'deepcad': {
                    'path': './data/deepcad',
                    'num_samples': 8046  # From paper
                },
                'fusion360': {
                    'path': './data/fusion360',
                    'num_samples': 1725  # From paper
                },
                'cc3d': {
                    'path': './data/cc3d',
                    'num_samples': 2973  # From paper
                }
            },
            'preprocessing': {
                'normalize_point_clouds': True,
                'add_noise': False,
                'noise_level': 0.01
            }
        })
        
        # Dataset generation configuration
        self.generation = NestedConfig({
            'max_sketches_per_model': 5,
            'max_extrudes_per_sketch': 3,
            'min_sketch_size': 0.1,
            'max_sketch_size': 10.0,
            'min_extrude_distance': 0.1,
            'max_extrude_distance': 5.0,
            'points_per_surface': 2048,
            'surface_sampling_method': 'uniform'
        })
        
        # Evaluation configuration
        self.evaluation = NestedConfig({
            'metrics': ['chamfer_distance', 'intersection_over_union', 'invalidity_ratio'],
            'batch_size': 16,
            'num_workers': 4,
            'voxel_resolution': 64,
            'results_dir': './results',
            'visualizations_dir': './visualizations'
        })
        
        # Hardware requirements
        self.hardware = NestedConfig({
            'recommended_gpu': 'NVIDIA H100',
            'min_gpu_memory': '24GB',
            'training_time_estimate': '12 hours'  # From paper
        })
        
        # Path configuration
        self.paths = NestedConfig({
            'pretrained_model_dir': './models/pretrained',
            'finetuned_model_dir': './models/finetuned',
            'raw_data_dir': './data/raw',
            'processed_data_dir': './data/processed',
            'output_dir': './output',
            'logs_dir': './logs'
        })
        
        # CadQuery configuration
        self.cadquery = NestedConfig({
            'max_code_length': 1000,
            'indentation': '    ',  # 4 spaces
            'execution_timeout': 30,  # seconds
            'max_retries': 3
        })
    
    def load_from_file(self, path: str) -> None:
        """
        Load configuration from YAML file and override defaults.
        
        Args:
            path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
        
        if yaml_config is None:
            warnings.warn("Configuration file is empty, using defaults")
            return
        
        # Update configuration with loaded values
        self._update_config(yaml_config)
        
        # Validate the updated configuration
        self._validate_config()
        
        # Update device after loading configuration
        self._device = self._detect_device()
        if hasattr(self.training, 'device'):
            self.training.device = self._device
    
    def _update_config(self, yaml_config: Dict[str, Any]) -> None:
        """
        Update configuration with values from YAML file.
        
        Args:
            yaml_config: Dictionary loaded from YAML file
        """
        for section_name, section_config in yaml_config.items():
            if hasattr(self, section_name):
                existing_section = getattr(self, section_name)
                if isinstance(existing_section, NestedConfig):
                    self._update_nested_config(existing_section, section_config)
                else:
                    setattr(self, section_name, section_config)
            else:
                # Create new section if it doesn't exist
                if isinstance(section_config, dict):
                    setattr(self, section_name, NestedConfig(section_config))
                else:
                    setattr(self, section_name, section_config)
    
    def _update_nested_config(self, existing_config: NestedConfig, new_config: Dict[str, Any]) -> None:
        """
        Recursively update nested configuration.
        
        Args:
            existing_config: Existing NestedConfig object
            new_config: New configuration dictionary
        """
        for key, value in new_config.items():
            if hasattr(existing_config, key) and isinstance(getattr(existing_config, key), NestedConfig):
                if isinstance(value, dict):
                    self._update_nested_config(getattr(existing_config, key), value)
                else:
                    setattr(existing_config, key, value)
            else:
                if isinstance(value, dict):
                    setattr(existing_config, key, NestedConfig(value))
                else:
                    setattr(existing_config, key, value)
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters for consistency and correctness.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate model configuration
        self._validate_model_config()
        
        # Validate training configuration
        self._validate_training_config()
        
        # Validate dataset configuration
        self._validate_dataset_config()
        
        # Validate generation configuration
        self._validate_generation_config()
        
        # Validate evaluation configuration
        self._validate_evaluation_config()
        
        # Validate paths
        self._validate_paths()
        
        # Validate CadQuery configuration
        self._validate_cadquery_config()
    
    def _validate_model_config(self) -> None:
        """Validate model configuration parameters."""
        # Validate projector parameters
        if self.model.projector.num_points <= 0:
            raise ValueError("num_points must be positive")
        if self.model.projector.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.model.projector.fourier_freqs <= 0:
            raise ValueError("fourier_freqs must be positive")
        
        # Validate generation parameters
        if self.model.generation.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.model.generation.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0 <= self.model.generation.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
    
    def _validate_training_config(self) -> None:
        """Validate training configuration parameters."""
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.training.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not (0 <= self.training.weight_decay <= 1):
            raise ValueError("weight_decay must be between 0 and 1")
        
        # Validate device
        if self.training.device not in ['cuda', 'cpu', 'auto']:
            if not self.training.device.startswith('cuda:'):
                raise ValueError("device must be 'cuda', 'cpu', 'auto', or 'cuda:N'")
    
    def _validate_dataset_config(self) -> None:
        """Validate dataset configuration parameters."""
        if self.dataset.train.size <= 0:
            raise ValueError("train dataset size must be positive")
        
        # Validate test dataset sample counts
        if self.dataset.test.deepcad.num_samples <= 0:
            raise ValueError("DeepCAD num_samples must be positive")
        if self.dataset.test.fusion360.num_samples <= 0:
            raise ValueError("Fusion360 num_samples must be positive")
        if self.dataset.test.cc3d.num_samples <= 0:
            raise ValueError("CC3D num_samples must be positive")
        
        # Validate preprocessing parameters
        if hasattr(self.dataset.preprocessing, 'noise_level'):
            if not (0 <= self.dataset.preprocessing.noise_level <= 1):
                raise ValueError("noise_level must be between 0 and 1")
    
    def _validate_generation_config(self) -> None:
        """Validate dataset generation configuration parameters."""
        if self.generation.max_sketches_per_model <= 0:
            raise ValueError("max_sketches_per_model must be positive")
        if self.generation.max_extrudes_per_sketch <= 0:
            raise ValueError("max_extrudes_per_sketch must be positive")
        
        # Validate size constraints
        if self.generation.min_sketch_size >= self.generation.max_sketch_size:
            raise ValueError("min_sketch_size must be less than max_sketch_size")
        if self.generation.min_extrude_distance >= self.generation.max_extrude_distance:
            raise ValueError("min_extrude_distance must be less than max_extrude_distance")
        
        if self.generation.points_per_surface <= 0:
            raise ValueError("points_per_surface must be positive")
    
    def _validate_evaluation_config(self) -> None:
        """Validate evaluation configuration parameters."""
        valid_metrics = {'chamfer_distance', 'intersection_over_union', 'invalidity_ratio'}
        for metric in self.evaluation.metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics: {valid_metrics}")
        
        if self.evaluation.batch_size <= 0:
            raise ValueError("evaluation batch_size must be positive")
        if self.evaluation.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.evaluation.voxel_resolution <= 0:
            raise ValueError("voxel_resolution must be positive")
        
        # Check if voxel resolution is power of 2 for efficiency
        if self.evaluation.voxel_resolution & (self.evaluation.voxel_resolution - 1) != 0:
            warnings.warn("voxel_resolution should be a power of 2 for optimal performance")
    
    def _validate_paths(self) -> None:
        """Validate and create necessary directories."""
        path_attrs = [
            'checkpoint_dir', 'results_dir', 'visualizations_dir',
            'pretrained_model_dir', 'finetuned_model_dir', 'output_dir', 'logs_dir'
        ]
        
        for attr in path_attrs:
            # Check training paths
            if hasattr(self.training, attr):
                path = Path(getattr(self.training, attr))
                path.mkdir(parents=True, exist_ok=True)
            
            # Check evaluation paths
            if hasattr(self.evaluation, attr):
                path = Path(getattr(self.evaluation, attr))
                path.mkdir(parents=True, exist_ok=True)
            
            # Check general paths
            if hasattr(self.paths, attr):
                path = Path(getattr(self.paths, attr))
                path.mkdir(parents=True, exist_ok=True)
    
    def _validate_cadquery_config(self) -> None:
        """Validate CadQuery configuration parameters."""
        if self.cadquery.max_code_length <= 0:
            raise ValueError("max_code_length must be positive")
        if self.cadquery.execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")
        if self.cadquery.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
    
    def _detect_device(self) -> str:
        """
        Detect available device for computation.
        