```python
"""
Dataset loader for CAD-Recode system.
Implements PyTorch Dataset interface for efficient loading of point cloud and CAD code pairs.
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
import numpy as np

from config import Config
from utils.cad_utils import CADUtils
from utils.point_cloud_utils import PointCloudUtils


class CADDataset(Dataset):
    """
    PyTorch Dataset for CAD-Recode training data.
    Handles point cloud and CAD code pairs with efficient loading and tokenization.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 tokenizer: PreTrainedTokenizer,
                 config: Optional[Config] = None,
                 split: str = 'train',
                 max_samples: Optional[int] = None):
        """
        Initialize CAD dataset.
        
        Args:
            data_dir: Directory containing point clouds and CAD codes
            tokenizer: Pre-trained tokenizer for CAD code processing
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load (for debugging)
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.config = config if config is not None else Config()
        self.split = split
        self.max_samples = max_samples
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize utilities
        self.cad_utils = CADUtils(self.config)
        
        # Dataset parameters from config
        self.max_length = self.config.model.generation.max_length
        self.num_points = self.config.model.projector.num_points
        self.normalize_point_clouds = self.config.dataset.preprocessing.normalize_point_clouds
        self.add_noise = self.config.dataset.preprocessing.add_noise
        self.noise_level = getattr(self.config.dataset.preprocessing, 'noise_level', 0.01)
        
        # Data storage
        self.point_clouds: List[str] = []  # Paths to point cloud files
        self.cad_codes: List[str] = []     # Paths to CAD code files
        self.valid_indices: List[int] = [] # Indices of valid samples
        
        # Performance optimization
        self._cache_size = 1000  # Number of samples to cache in memory
        self._sample_cache: Dict[int, Tuple[torch.Tensor, str]] = {}
        
        # Load dataset
        self._load_dataset()
        
        self.logger.info(f"Initialized CADDataset: {len(self.valid_indices)} valid samples "
                        f"from {data_dir} ({split} split)")
    
    def _load_dataset(self) -> None:
        """Load and validate dataset files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        # Define subdirectories
        point_clouds_dir = self.data_dir / "point_clouds"
        cad_codes_dir = self.data_dir / "cad_codes"
        
        if not point_clouds_dir.exists():
            raise FileNotFoundError(f"Point clouds directory not found: {point_clouds_dir}")
        if not cad_codes_dir.exists():
            raise FileNotFoundError(f"CAD codes directory not found: {cad_codes_dir}")
        
        # Find all point cloud files
        pc_files = list(point_clouds_dir.glob("*.pt"))
        if not pc_files:
            # Try alternative formats
            pc_files = list(point_clouds_dir.glob("*.npy"))
            if not pc_files:
                pc_files = list(point_clouds_dir.glob("*.npz"))
        
        if not pc_files:
            raise ValueError(f"No point cloud files found in {point_clouds_dir}")
        
        # Sort files for consistent ordering
        pc_files.sort()
        
        # Limit samples if specified
        if self.max_samples is not None:
            pc_files = pc_files[:self.max_samples]
        
        # Validate file pairs
        valid_count = 0
        invalid_count = 0
        
        for pc_file in pc_files:
            # Extract sample ID from filename
            sample_id = pc_file.stem
            
            # Find corresponding CAD code file
            cad_file = None
            for ext in ['.py', '.txt']:
                candidate = cad_codes_dir / f"{sample_id}{ext}"
                if candidate.exists():
                    cad_file = candidate
                    break
            
            if cad_file is None:
                self.logger.debug(f"No CAD code file found for {pc_file.name}")
                invalid_count += 1
                continue
            
            # Validate files
            if self._validate_sample_files(pc_file, cad_file):
                self.point_clouds.append(str(pc_file))
                self.cad_codes.append(str(cad_file))
                self.valid_indices.append(valid_count)
                valid_count += 1
            else:
                invalid_count += 1
        
        if valid_count == 0:
            raise ValueError("No valid samples found in dataset")
        
        self.logger.info(f"Dataset validation: {valid_count} valid, {invalid_count} invalid samples")
        
        # Log statistics
        if invalid_count > 0:
            invalid_ratio = invalid_count / (valid_count + invalid_count)
            if invalid_ratio > 0.1:  # More than 10% invalid
                self.logger.warning(f"High invalid sample ratio: {invalid_ratio:.2%}")
    
    def _validate_sample_files(self, pc_file: Path, cad_file: Path) -> bool:
        """
        Validate individual sample files.
        
        Args:
            pc_file: Path to point cloud file
            cad_file: Path to CAD code file
            
        Returns:
            bool: True if both files are valid
        """
        try:
            # Check file sizes
            if pc_file.stat().st_size == 0:
                self.logger.debug(f"Empty point cloud file: {pc_file.name}")
                return False
            
            if cad_file.stat().st_size == 0:
                self.logger.debug(f"Empty CAD code file: {cad_file.name}")
                return False
            
            # Quick point cloud validation
            try:
                if pc_file.suffix == '.pt':
                    pc_data = torch.load(pc_file, map_location='cpu')
                elif pc_file.suffix == '.npy':
                    pc_data = torch.from_numpy(np.load(pc_file))
                elif pc_file.suffix == '.npz':
                    data = np.load(pc_file)
                    if 'points' in data:
                        pc_data = torch.from_numpy(data['points'])
                    else:
                        pc_data = torch.from_numpy(data[list(data.keys())[0]])
                else:
                    return False
                
                # Validate point cloud shape
                if not isinstance(pc_data, torch.Tensor):
                    return False
                if pc_data.ndim != 2 or pc_data.shape[1] != 3:
                    return False
                if len(pc_data) == 0:
                    return False
                if not torch.isfinite(pc_data).all():
                    return False
                    
            except Exception as e:
                self.logger.debug(f"Point cloud validation failed for {pc_file.name}: {e}")
                return False
            
            # Quick CAD code validation
            try:
                with open(cad_file, 'r', encoding='utf-8') as f:
                    code_content = f.read().strip()
                
                if not code_content:
                    return False
                
                # Basic syntax validation
                if not self.cad_utils.validate_cad_code(code_content):
                    self.logger.debug(f"CAD code validation failed for {cad_file.name}")
                    return False
                    
            except Exception as e:
                self.logger.debug(f"CAD code validation failed for {cad_file.name}: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Sample validation failed for {pc_file.name}: {e}")
            return False
    
    def _load_point_cloud(self, pc_path: str) -> torch.Tensor:
        """
        Load point cloud from file.
        
        Args:
            pc_path: Path to point cloud file
            
        Returns:
            torch.Tensor: Point cloud with shape [N, 3]
            
        Raises:
            ValueError: If loading fails
        """
        try:
            pc_file = Path(pc_path)
            
            if pc_file.suffix == '.pt':
                point_cloud = torch.load(pc_file, map_location='cpu')
            elif pc_file.suffix == '.npy':
                point_cloud = torch.from_numpy(np.load(pc_file).astype(np.float32))
            elif pc_file.suffix == '.npz':
                data = np.load(pc_file)
                if 'points' in data:
                    point_cloud = torch.from_numpy(data['points'].astype(np.float32))
                elif 'point_cloud' in data:
                    point_cloud = torch.from_numpy(data['point_cloud'].astype(np.float32))
                else:
                    # Use first array
                    key = list(data.keys())[0]
                    point_cloud = torch.from_numpy(data[key].astype(np.float32))
            else:
                raise ValueError(f"Unsupported point cloud format: {pc_file.suffix}")
            
            # Validate shape
            if point_cloud.ndim == 1:
                point_cloud = point_cloud.reshape(-1, 3)
            
            if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
                raise ValueError(f"Invalid point cloud shape: {point_cloud.shape}")
            
            if len(point_cloud) == 0:
                raise ValueError("Empty point cloud")
            
            # Check for invalid coordinates
            if not torch.isfinite(point_cloud).all():
                self.logger.warning(f"Point cloud contains invalid coordinates: {pc_path}")
                # Filter out invalid points
                valid_mask = torch.isfinite(point_cloud).all(dim=1)
                point_cloud = point_cloud[valid_mask]
                
                if len(point_cloud) == 0:
                    raise ValueError("No valid points after filtering")
            
            return point_cloud
            
        except Exception as e:
            raise ValueError(f"Failed to load point cloud from {pc_path}: {e}")
    
    def _load_cad_code(self, code_path: str) -> str:
        """
        Load CAD code from file.
        
        Args:
            code_path: Path to CAD code file
            
        Returns:
            str: CAD code content
            
        Raises:
            ValueError: If loading fails
        """
        try:
            with open(code_path, 'r', encoding='utf-8') as f:
                code_content = f.read().strip()
            
            if not code_content:
                raise ValueError("Empty CAD code file")
            
            return code_content
            
        except Exception as e:
            raise ValueError(f"Failed to load CAD code from {code_path}: {e}")
    
    def _preprocess_point_cloud(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Preprocess point cloud according to configuration.
        
        Args:
            point_cloud: Raw point cloud tensor
            
        Returns:
            torch.Tensor: Preprocessed point cloud
        """
        # Apply furthest point sampling to get exactly num_points
        if len(point_cloud) != self.num_points:
            point_cloud = PointCloudUtils.furthest_point_sampling(point_cloud, self.num_points)
        
        # Normalize coordinates if enabled
        if self.normalize_point_clouds:
            point_cloud = PointCloudUtils.normalize_point_cloud(point_cloud)
        
        # Add noise for data augmentation if enabled (only during training)
        if self.add_noise and self.split == 'train':
            point_cloud = PointCloudUtils.add_noise(point_cloud, self.noise_level)
        
        return point_cloud
    
    def _tokenize_cad_code(self, code: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize CAD code using the pre-trained tokenizer.
        
        Args:
            code: CAD code string
            
        Returns:
            Dict[str, torch.Tensor]: Tokenized code with input_ids and attention_mask
        """
        try:
            # Tokenize with truncation and padding
            tokenized = self.tokenizer(
                code,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            
            # Remove batch dimension added by tokenizer
            input_ids = tokenized['input_ids'].squeeze(0)  # [seq_len]
            attention_mask = tokenized['attention_mask'].squeeze(0)  # [seq_len]
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
        except Exception as e:
            self.logger.error(f"Tokenization failed for code: {code[:100]}... Error: {e}")
            # Return empty tokenization as fallback
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }
    
    def __len__(self) -> int:
        """Return number of valid samples in dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (point_cloud, input_ids, attention_mask)
        """
        if idx < 0 or idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self.valid_indices)})")
        
        # Check cache first
        if idx in self._sample_cache:
            cached_pc, cached_code = self._sample_cache[idx]
            tokenized = self._tokenize_cad_code(cached_code)
            return cached_pc, tokenized['input_ids'], tokenized['attention_mask']
        
        try:
            # Load data
            pc_path = self.point_clouds[idx]
            code_path = self.cad_codes[idx]
            
            point_cloud = self._load_point_cloud(pc_path)
            cad_code = self._load_cad_code(code_path)
            
            # Preprocess point cloud
            point_cloud = self._preprocess_point_cloud(point_cloud)
            
            # Cache sample if cache is not full
            if len(self._sample_cache) < self._cache_size:
                self._sample_cache[idx] = (point_cloud.clone(), cad_code)
            
            # Tokenize CA