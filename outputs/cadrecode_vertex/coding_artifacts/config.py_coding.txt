```python
"""
Configuration management for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module provides centralized configuration management for all hyperparameters,
model settings, and experimental configurations specified in the paper.
"""

import os
import yaml
import warnings
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class Config:
    """
    Central configuration management class for CAD-Recode.
    
    Handles all hyperparameters, model settings, and experimental configurations
    as specified in the paper, ensuring reproducibility and parameter validation.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with paper-specified defaults.
        
        Args:
            config_dict: Optional dictionary to override default values
        """
        # Initialize with paper-specified defaults
        self._config = self._get_paper_defaults()
        
        # Override with provided configuration if given
        if config_dict is not None:
            self._deep_update(self._config, config_dict)
        
        # Validate configuration parameters
        self.validate_parameters()
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance with loaded parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML file is malformed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            warnings.warn(f"Config file {config_path} not found. Using paper defaults.")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                warnings.warn("Empty config file. Using paper defaults.")
                return cls()
                
            return cls(config_dict)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading config file: {e}")
    
    def validate_parameters(self) -> bool:
        """
        Validate configuration parameters against paper specifications.
        
        Returns:
            True if all parameters are valid
            
        Raises:
            ValueError: If critical parameters are invalid
        """
        # Validate critical training parameters from paper Section 4.3 & Appendix A
        training_config = self.get_training_config()
        
        # Learning rate must be 0.0002 for reproducibility
        if training_config['learning_rate'] != 0.0002:
            warnings.warn(f"Learning rate {training_config['learning_rate']} differs from paper value 0.0002")
        
        # Batch size must be 18 (paper-specified for H100 GPU)
        if training_config['batch_size'] != 18:
            warnings.warn(f"Batch size {training_config['batch_size']} differs from paper value 18")
        
        # Weight decay must be 0.01
        if training_config['weight_decay'] != 0.01:
            warnings.warn(f"Weight decay {training_config['weight_decay']} differs from paper value 0.01")
        
        # Validate model parameters
        model_config = self.get_model_config()
        
        # Embedding dimension must be 1536 (d_q from paper)
        if model_config['embedding_dim'] != 1536:
            raise ValueError(f"Embedding dimension must be 1536, got {model_config['embedding_dim']}")
        
        # Number of points must be 256 (n_p from paper)
        if model_config['num_points'] != 256:
            warnings.warn(f"Number of points {model_config['num_points']} differs from paper value 256")
        
        # Validate data parameters
        data_config = self.get_data_config()
        
        # Coordinate range validation
        coord_range = data_config['procedural_dataset']['coordinate_range']
        if coord_range != [-100, 100]:
            warnings.warn(f"Coordinate range {coord_range} differs from paper value [-100, 100]")
        
        # Primitives range validation
        prim_range = data_config['procedural_dataset']['num_primitives_range']
        if prim_range != [3, 8]:
            warnings.warn(f"Primitives range {prim_range} differs from paper value [3, 8]")
        
        # Validate evaluation parameters
        eval_config = self.get_evaluation_config()
        
        # Number of candidates must be 10 (test-time sampling from paper)
        if eval_config['num_candidates'] != 10:
            warnings.warn(f"Number of candidates {eval_config['num_candidates']} differs from paper value 10")
        
        # Chamfer distance points must be 8192
        cd_points = eval_config['metrics']['chamfer_distance']['num_points']
        if cd_points != 8192:
            warnings.warn(f"Chamfer distance points {cd_points} differs from paper value 8192")
        
        return True
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Returns:
            Dictionary containing model configuration parameters
        """
        return self._config['model'].copy()
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training-specific configuration.
        
        Returns:
            Dictionary containing training configuration parameters
        """
        return self._config['training'].copy()
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data-specific configuration.
        
        Returns:
            Dictionary containing data configuration parameters
        """
        return self._config['data'].copy()
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation-specific configuration.
        
        Returns:
            Dictionary containing evaluation configuration parameters
        """
        return self._config['evaluation'].copy()
    
    def get_system_config(self) -> Dict[str, Any]:
        """
        Get system-specific configuration.
        
        Returns:
            Dictionary containing system configuration parameters
        """
        return self._config['system'].copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging-specific configuration.
        
        Returns:
            Dictionary containing logging configuration parameters
        """
        return self._config['logging'].copy()
    
    def get_paths_config(self) -> Dict[str, Any]:
        """
        Get paths configuration.
        
        Returns:
            Dictionary containing path configuration parameters
        """
        return self._config['paths'].copy()
    
    def get_cadquery_config(self) -> Dict[str, Any]:
        """
        Get CadQuery-specific configuration.
        
        Returns:
            Dictionary containing CadQuery configuration parameters
        """
        return self._config['cadquery'].copy()
    
    def get_ablation_config(self) -> Dict[str, Any]:
        """
        Get ablation study configuration.
        
        Returns:
            Dictionary containing ablation study parameters
        """
        return self._config['ablation'].copy()
    
    def get_hyperparameter(self, section: str, param: str, default: Any = None) -> Any:
        """
        Safely access a specific hyperparameter.
        
        Args:
            section: Configuration section name
            param: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        try:
            return self._config[section][param]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(f"Parameter '{param}' not found in section '{section}'")
    
    def update_parameter(self, section: str, param: str, value: Any) -> None:
        """
        Update a specific parameter with validation.
        
        Args:
            section: Configuration section name
            param: Parameter name
            value: New parameter value
        """
        if section not in self._config:
            self._config[section] = {}
        
        old_value = self._config[section].get(param)
        self._config[section][param] = value
        
        # Re-validate after update
        try:
            self.validate_parameters()
        except (ValueError, Warning) as e:
            # Restore old value if validation fails
            if old_value is not None:
                self._config[section][param] = old_value
            raise e
    
    def export_config(self, path: str) -> None:
        """
        Export current configuration to YAML file.
        
        Args:
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def create_ablation_configs(self, ablation_params: Dict[str, List]) -> List['Config']:
        """
        Generate configurations for ablation studies.
        
        Args:
            ablation_params: Dictionary with parameter names and value lists
            
        Returns:
            List of Config instances for ablation study
        """
        configs = []
        base_config = self._config.copy()
        
        # Handle point cloud size ablation
        if 'point_cloud_sizes' in ablation_params:
            for size in ablation_params['point_cloud_sizes']:
                config_dict = self._deep_copy(base_config)
                config_dict['model']['num_points'] = size
                configs.append(Config(config_dict))
        
        # Handle model size ablation
        if 'model_sizes' in ablation_params:
            for model_name in ablation_params['model_sizes']:
                config_dict = self._deep_copy(base_config)
                config_dict['model']['llm_model_name'] = model_name
                configs.append(Config(config_dict))
        
        # Handle dataset size ablation
        if 'dataset_sizes' in ablation_params:
            for size in ablation_params['dataset_sizes']:
                config_dict = self._deep_copy(base_config)
                config_dict['data']['procedural_dataset']['size'] = size
                configs.append(Config(config_dict))
        
        return configs
    
    def _get_paper_defaults(self) -> Dict[str, Any]:
        """
        Get all paper-specified default values.
        
        Returns:
            Dictionary with complete default configuration
        """
        return {
            'model': {
                'llm_model_name': 'Qwen/Qwen2-1.5B',
                'embedding_dim': 1536,
                'num_points': 256,
                'fourier_encoding': {
                    'num_freqs': 64
                },
                'start_token': '<s>',
                'end_token': '<e>'
            },
            'training': {
                'learning_rate': 0.0002,
                'weight_decay': 0.01,
                'batch_size': 18,
                'num_iterations': 100000,
                'warmup_steps': 1000,
                'optimizer': 'AdamW',
                'scheduler': 'cosine',
                'noise_probability': 0.5,
                'noise_std': 0.01,
                'training_mode': 'end_to_end',
                'loss_function': 'nll'
            },
            'data': {
                'procedural_dataset': {
                    'size': 1000000,
                    'num_primitives_range': [3, 8],
                    'coordinate_range': [-100, 100],
                    'coordinate_resolution': 1
                },
                'datasets': {
                    'deepcad': {
                        'test_size': 8046,
                        'train_size': 160000
                    },
                    'fusion360': {
                        'test_size': 1725
                    },
                    'cc3d': {
                        'test_size': 2973
                    }
                },
                'point_cloud': {
                    'num_sample_points': 8192,
                    'normalization': 'unit_box'
                }
            },
            'evaluation': {
                'num_candidates': 10,
                'sampling_strategy': 'different_point_cloud_sampling',
                'metrics': {
                    'chamfer_distance': {
                        'num_points': 8192,
                        'scale_factor': 1000
                    },
                    'intersection_over_union': {
                        'compute_from': 'meshes'
                    },
                    'invalidity_ratio': {
                        'check_syntax': True,
                        'check_cad_semantics': True
                    }
                }
            },
            'system': {
                'device': 'cuda',
                'mixed_precision': False,
                'num_workers': 4,
                'pin_memory': True
            },
            'logging': {
                'log_interval': 100,
                'save_interval': 5000,
                'wandb_project': 'cad-recode'
            },
            'paths': {
                'data_dir': './data',
                'checkpoints_dir': './checkpoints',
                'results_dir': './results',
                'logs_dir': './logs'
            },
            'cadquery': {
                'library_import': 'import cadquery as cq',
                'validation_timeout': 30
            },
            'ablation': {
                'point_cloud_sizes': [64, 128, 256],
                'model_sizes': ['Qwen/Qwen2-0.5B', 'Qwen/Qwen2-1.5B'],
                'dataset_sizes': [160000, 1000000]
            }
        }
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update target dictionary with source dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _deep_copy(self, obj: Any) -> Any:
        """
        Create a deep copy of an object.
        
        Args:
            obj: Object to copy
            
        Returns:
            Deep copy of the object
        """
        import copy
        return copy.deepcopy(obj)
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to configuration.
        
        Args:
            key: Configuration section key
            
        Returns:
            Configuration section
        """
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style setting of configuration.
        
        Args:
            key: Configuration section key
            value: Configuration section value
        """
        self._config[key] = value
        self.validate_parameters()
    
    def __contains__(self, key: str) -> bool:
        """
        Check if configuration section exists.
        
        Args:
            key: Configuration section key
            
        Returns:
            True if section exists
        """
        return key in self._config
    
    def keys(self):
        """Get configuration section keys."""
        return self._config.keys()
    
    def values(self):
        """Get configuration section values."""
        return self._config.values()
    
    def items(self):
        """Get configuration section items."""
        