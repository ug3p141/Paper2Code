```python
"""
Dataset loading and preprocessing for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module handles loading and preprocessing of all datasets (procedural, DeepCAD, 
Fusion360, CC3D) with unified point cloud preprocessing, tokenization, and PyTorch 
Dataset/DataLoader creation. Implements the complete data pipeline from raw data 
sources to model-ready batches.
"""

import os
import sys
import json
import pickle
import logging
import warnings
import h5py
import tempfile
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator, Callable
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import trimesh
import open3d as o3d

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Tokenization will be limited.")

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    warnings.warn("CadQuery not available. CAD validation will be limited.")

# Import utilities
from utils.point_cloud_utils import (
    furthest_point_sampling, 
    add_gaussian_noise, 
    normalize_point_cloud,
    mesh_to_point_cloud,
    validate_point_cloud
)
from utils.cad_validation import CADValidator
from data.dataset_generator import DatasetGenerator


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Data structure for individual dataset samples."""
    point_cloud: np.ndarray  # Shape: (N, 3)
    cad_code: str
    metadata: Dict[str, Any]
    dataset_name: str
    sample_id: str


@dataclass
class DatasetStats:
    """Statistics for dataset loading and preprocessing."""
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    preprocessing_failures: int = 0
    validation_failures: int = 0
    loading_time: float = 0.0
    avg_points_per_sample: float = 0.0
    avg_code_length: float = 0.0


class CADDataset(Dataset):
    """
    PyTorch Dataset for CAD-Recode training and evaluation.
    
    Handles point cloud preprocessing, code tokenization, and data augmentation
    with unified interface across all dataset types.
    """
    
    def __init__(
        self,
        samples: List[DataSample],
        tokenizer: Any,
        config: Dict[str, Any],
        is_training: bool = False,
        cache_size: int = 1000
    ):
        """
        Initialize CAD dataset.
        
        Args:
            samples: List of preprocessed data samples
            tokenizer: HuggingFace tokenizer for code tokenization
            config: Configuration dictionary
            is_training: Whether this is a training dataset (enables augmentation)
            cache_size: Size of LRU cache for preprocessed samples
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
        
        # Extract configuration
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        
        self.num_points = model_config.get('num_points', 256)
        self.start_token = model_config.get('start_token', '<s>')
        self.end_token = model_config.get('end_token', '<e>')
        
        # Data augmentation settings (only for training)
        self.noise_probability = training_config.get('noise_probability', 0.5) if is_training else 0.0
        self.noise_std = training_config.get('noise_std', 0.01)
        
        # Point cloud preprocessing settings
        pc_config = data_config.get('point_cloud', {})
        self.normalization_method = pc_config.get('normalization', 'unit_box')
        
        # Initialize cache for preprocessed samples
        self._cache = {}
        self._cache_size = cache_size
        self._cache_access_order = []
        self._cache_lock = threading.Lock()
        
        # Precompute tokenization for all samples
        self._precompute_tokenization()
        
        logger.info(f"CADDataset initialized:")
        logger.info(f"  Samples: {len(self.samples)}")
        logger.info(f"  Training mode: {self.is_training}")
        logger.info(f"  Augmentation: noise_prob={self.noise_probability}")
        logger.info(f"  Target points: {self.num_points}")
    
    def _precompute_tokenization(self) -> None:
        """Precompute tokenization for all code samples."""
        logger.info("Precomputing tokenization for all samples...")
        
        for i, sample in enumerate(self.samples):
            try:
                # Add special tokens
                code_with_tokens = f"{self.start_token} {sample.cad_code} {self.end_token}"
                
                # Tokenize
                tokens = self.tokenizer(
                    code_with_tokens,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512  # Reasonable max length for CAD code
                )
                
                # Store tokenized data in sample metadata
                sample.metadata['input_ids'] = tokens['input_ids'].squeeze(0)
                sample.metadata['attention_mask'] = tokens['attention_mask'].squeeze(0)
                
                # Create labels for training (shift input_ids by 1)
                if self.is_training:
                    labels = sample.metadata['input_ids'].clone()
                    # Mask start token in labels
                    labels[0] = -100
                    sample.metadata['labels'] = labels
                
            except Exception as e:
                logger.error(f"Failed to tokenize sample {i}: {e}")
                # Mark sample as invalid
                sample.metadata['tokenization_failed'] = True
        
        # Filter out samples with tokenization failures
        valid_samples = [s for s in self.samples if not s.metadata.get('tokenization_failed', False)]
        if len(valid_samples) < len(self.samples):
            logger.warning(f"Removed {len(self.samples) - len(valid_samples)} samples due to tokenization failures")
            self.samples = valid_samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with preprocessing.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing preprocessed sample data
        """
        # Check cache first
        with self._cache_lock:
            if idx in self._cache:
                # Update access order for LRU
                self._cache_access_order.remove(idx)
                self._cache_access_order.append(idx)
                return self._cache[idx].copy()
        
        # Process sample
        sample = self.samples[idx]
        
        try:
            # Preprocess point cloud
            point_cloud = self._preprocess_point_cloud(sample.point_cloud.copy())
            
            # Get tokenized data
            input_ids = sample.metadata['input_ids']
            attention_mask = sample.metadata['attention_mask']
            
            # Prepare output dictionary
            output = {
                'point_cloud': torch.from_numpy(point_cloud).float(),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'raw_code': sample.cad_code,
                'dataset_name': sample.dataset_name,
                'sample_id': sample.sample_id
            }
            
            # Add labels for training
            if self.is_training and 'labels' in sample.metadata:
                output['labels'] = sample.metadata['labels']
            
            # Add metadata
            output['metadata'] = {
                'original_points': sample.point_cloud.shape[0],
                'code_length': len(sample.cad_code),
                'dataset_name': sample.dataset_name
            }
            
            # Cache the result
            with self._cache_lock:
                self._add_to_cache(idx, output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return a dummy sample to avoid breaking training
            return self._get_dummy_sample()
    
    def _preprocess_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Preprocess point cloud with normalization, sampling, and augmentation.
        
        Args:
            point_cloud: Input point cloud array of shape (N, 3)
            
        Returns:
            Preprocessed point cloud array of shape (num_points, 3)
        """
        # Convert to torch tensor for processing
        pc_tensor = torch.from_numpy(point_cloud).float()
        
        # Validate input
        is_valid, error_msg = validate_point_cloud(pc_tensor, min_points=1)
        if not is_valid:
            logger.warning(f"Invalid point cloud: {error_msg}. Using dummy data.")
            pc_tensor = torch.randn(self.num_points, 3)
        
        # Normalize point cloud
        normalized_pc, _ = normalize_point_cloud(pc_tensor, method=self.normalization_method)
        
        # Apply furthest point sampling
        if normalized_pc.shape[0] >= self.num_points:
            sampled_pc = furthest_point_sampling(normalized_pc, self.num_points)
        else:
            # Handle case with fewer points than target
            logger.warning(f"Point cloud has {normalized_pc.shape[0]} points, less than target {self.num_points}")
            # Repeat points to reach target size
            repeat_factor = (self.num_points + normalized_pc.shape[0] - 1) // normalized_pc.shape[0]
            repeated_pc = normalized_pc.repeat(repeat_factor, 1)
            sampled_pc = repeated_pc[:self.num_points]
        
        # Apply data augmentation (only during training)
        if self.is_training and self.noise_probability > 0:
            sampled_pc = add_gaussian_noise(
                sampled_pc, 
                noise_std=self.noise_std, 
                probability=self.noise_probability
            )
        
        return sampled_pc.numpy()
    
    def _add_to_cache(self, idx: int, data: Dict[str, torch.Tensor]) -> None:
        """Add sample to LRU cache."""
        # Remove oldest item if cache is full
        if len(self._cache) >= self._cache_size:
            oldest_idx = self._cache_access_order.pop(0)
            del self._cache[oldest_idx]
        
        # Add new item
        self._cache[idx] = data.copy()
        self._cache_access_order.append(idx)
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Generate dummy sample for error cases."""
        dummy_code = f"{self.start_token} import cadquery as cq; r = cq.Workplane().box(1, 1, 1) {self.end_token}"
        
        # Tokenize dummy code
        tokens = self.tokenizer(
            dummy_code,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512
        )
        
        return {
            'point_cloud': torch.randn(self.num_points, 3),
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'raw_code': dummy_code,
            'dataset_name': 'dummy',
            'sample_id': 'dummy_sample',
            'metadata': {
                'original_points': self.num_points,
                'code_length': len(dummy_code),
                'dataset_name': 'dummy'
            }
        }


class DatasetLoader:
    """
    Central dataset loading and preprocessing coordinator.
    
    Handles loading from multiple data sources (procedural, DeepCAD, Fusion360, CC3D)
    with unified preprocessing pipeline and PyTorch DataLoader creation.
    """
    
    def __init__(self, config: Dict[str, Any], tokenizer: Any = None):
        """
        Initialize dataset loader.
        
        Args:
            config: Configuration dictionary from config.yaml
            tokenizer: HuggingFace tokenizer (will be loaded if None)
        """
        self.config = config
        
        # Initialize tokenizer
        if tokenizer is None:
            model_config = config.get('model', {})
            model_name = model_config.get('llm_model_name', 'Qwen/Qwen2-1.5B')
            
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    # Add special tokens if they don't exist
                    special_tokens = {
                        'additional_special_tokens': [
                            model_config.get('start_token', '<s>'),
                            model_config.get('end_token', '<e>')
                        ]
                    }
                    self.tokenizer.add_special_tokens(special_tokens)
                except Exception as e:
                    logger.error(f"Failed to load tokenizer: {e}")
                    self.tokenizer = None
            else:
                logger.warning("Transformers not available, tokenizer disabled")
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer
        
        # Extract configuration
        self.data_config = config.get('data', {})
        self.system_config = config.get('system', {})
        self.paths_config = config.get('paths', {})
        
        # Set up paths
        self.data_dir = Path(self.paths_config.get('data_dir', './data'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validator
        self.validator = CADValidator(config.get('cadquery', {}))
        
        # Initialize dataset generator for procedural data
        self.dataset_generator = DatasetGenerator(config)
        
        # Statistics tracking
        self.stats = {
            'procedural': DatasetStats(),
            'deepcad': DatasetStats(),
            'fusion360': DatasetStats(),
            'cc3d': DatasetStats()
        }
        
        logger.info(f"DatasetLoader initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Tokenizer: {'Available' if self.tokenizer else 'Not available'}")
    
    def load_procedural_data(
        self, 
        split: str = 'train',
        num_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load procedural dataset generated by DatasetGenerator.
        
        Args:
            split: Dataset split ('train' or 'val')
            num_samples: Number of samples to load (None for all)
            
        Returns:
            CADDataset instance with procedural data
        """
        logger.info(f"Loading procedural dataset (split: {split})...")
        start_time = time.time()
        
        # Determine number of samples
        proc_config = self.data_config.get('procedural_dataset', {})
        total_size = proc_config.get('size', 1000000)
        
        if num_samples is None: