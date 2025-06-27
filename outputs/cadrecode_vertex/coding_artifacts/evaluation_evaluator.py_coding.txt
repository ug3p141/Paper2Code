```python
"""
Comprehensive evaluation pipeline for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the evaluation pipeline from Section 5.1 of the paper, including
test-time sampling strategy (Section 4.3) and multi-dataset evaluation across DeepCAD,
Fusion360, and CC3D datasets. The evaluator generates multiple candidates per input and
computes all three core metrics: Chamfer Distance, IoU, and Invalidity Ratio.
"""

import os
import sys
import json
import time
import logging
import warnings
import threading
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict
import traceback
import gc

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    warnings.warn("Trimesh not available. Mesh operations will be limited.")

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    warnings.warn("CadQuery not available. CAD execution will be limited.")

# Import project modules
from models.cad_recode_model import CADRecodeModel
from evaluation.metrics import Metrics
from utils.cad_validation import CADValidator
from utils.point_cloud_utils import furthest_point_sampling, add_gaussian_noise


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Data structure for individual evaluation samples."""
    sample_id: str
    point_cloud: np.ndarray
    gt_code: str
    gt_mesh: Any
    dataset_name: str
    metadata: Dict[str, Any]


@dataclass
class CandidateResult:
    """Result structure for individual candidate evaluation."""
    code: str
    is_valid: bool
    execution_time: float
    cad_model: Optional[Any] = None
    validation_error: Optional[str] = None


@dataclass
class SampleResult:
    """Result structure for individual sample evaluation."""
    sample_id: str
    dataset_name: str
    candidates: List[CandidateResult]
    selected_candidate: Optional[str]
    chamfer_distance: Optional[float]
    iou: Optional[float]
    is_invalid: bool
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class DatasetResults:
    """Result structure for complete dataset evaluation."""
    dataset_name: str
    num_samples: int
    num_valid: int
    mean_cd: float
    median_cd: float
    mean_iou: float
    invalidity_ratio: float
    sample_results: List[SampleResult]
    processing_time: float
    statistics: Dict[str, Any]


class Evaluator:
    """
    Comprehensive evaluation pipeline implementing test-time sampling and multi-dataset evaluation.
    
    This class coordinates the complete evaluation workflow:
    1. Generate multiple CAD code candidates using test-time sampling
    2. Validate and select best candidates
    3. Compute metrics (Chamfer Distance, IoU, Invalidity Ratio)
    4. Aggregate results across datasets for comparison with baselines
    """
    
    def __init__(
        self,
        model: CADRecodeModel,
        test_loaders: Dict[str, Any],
        config: Dict[str, Any]
    ):
        """
        Initialize evaluator with model, test data loaders, and configuration.
        
        Args:
            model: Trained CADRecodeModel for inference
            test_loaders: Dictionary of test data loaders keyed by dataset name
            config: Configuration dictionary from config.yaml
        """
        # Store core components
        self.model = model
        self.test_loaders = test_loaders
        self.config = config
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Extract evaluation configuration
        eval_config = config.get('evaluation', {})
        self.num_candidates = eval_config.get('num_candidates', 10)
        self.sampling_strategy = eval_config.get('sampling_strategy', 'different_point_cloud_sampling')
        
        # Extract model configuration
        model_config = config.get('model', {})
        self.num_points = model_config.get('num_points', 256)
        self.max_code_length = model_config.get('max_code_length', 512)
        
        # Extract system configuration
        system_config = config.get('system', {})
        self.device = system_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = system_config.get('num_workers', 4)
        
        # Extract paths configuration
        paths_config = config.get('paths', {})
        self.results_dir = Path(paths_config.get('results_dir', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics = Metrics(config)
        
        # Initialize validator
        cadquery_config = config.get('cadquery', {})
        self.validator = CADValidator(cadquery_config)
        
        # Evaluation settings
        self.generation_timeout = cadquery_config.get('validation_timeout', 30)
        self.enable_noise_augmentation = config.get('training', {}).get('noise_probability', 0.0) > 0
        self.noise_std = config.get('training', {}).get('noise_std', 0.01)
        
        # Result storage
        self.dataset_results: Dict[str, DatasetResults] = {}
        self.evaluation_stats = {
            'total_samples': 0,
            'total_candidates_generated': 0,
            'total_valid_candidates': 0,
            'total_execution_failures': 0,
            'total_validation_failures': 0,
            'evaluation_start_time': None,
            'evaluation_end_time': None
        }
        
        # Thread safety
        self._result_lock = threading.Lock()
        
        logger.info(f"Evaluator initialized:")
        logger.info(f"  Test datasets: {list(test_loaders.keys())}")
        logger.info(f"  Candidates per sample: {self.num_candidates}")
        logger.info(f"  Sampling strategy: {self.sampling_strategy}")
        logger.info(f"  Target points: {self.num_points}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Results directory: {self.results_dir}")
    
    def generate_multiple_candidates(
        self, 
        point_cloud: torch.Tensor, 
        num_candidates: int = 10
    ) -> List[str]:
        """
        Generate multiple CAD code candidates using test-time sampling strategy.
        
        Implements the test-time sampling approach from Section 4.3:
        - Generate num_candidates distinct codes
        - Each uses different random sampling of input point cloud
        - Optional noise augmentation for diversity
        
        Args:
            point_cloud: Input point cloud tensor of shape (N, 3)
            num_candidates: Number of candidates to generate (default: 10)
            
        Returns:
            List of generated CAD code strings
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If generation fails
        """
        # Input validation
        if not isinstance(point_cloud, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(point_cloud)}")
        
        if point_cloud.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D tensor, got {point_cloud.dim()}D")
        
        if point_cloud.size(-1) != 3:
            raise ValueError(f"Expected 3D coordinates, got {point_cloud.size(-1)}D")
        
        # Ensure point cloud is 2D (N, 3)
        if point_cloud.dim() == 3:
            if point_cloud.size(0) != 1:
                raise ValueError(f"Expected batch size 1 for 3D input, got {point_cloud.size(0)}")
            point_cloud = point_cloud.squeeze(0)
        
        # Move to correct device
        point_cloud = point_cloud.to(self.device)
        
        candidates = []
        generation_times = []
        
        try:
            with torch.no_grad():
                for candidate_idx in range(num_candidates):
                    candidate_start_time = time.time()
                    
                    try:
                        # Apply different sampling for each candidate
                        sampled_pc = self._apply_candidate_sampling(
                            point_cloud, 
                            candidate_idx, 
                            num_candidates
                        )
                        
                        # Add batch dimension for model input
                        batch_pc = sampled_pc.unsqueeze(0)  # Shape: (1, num_points, 3)
                        
                        # Generate CAD code
                        generated_code = self.model.generate_code(
                            point_cloud=batch_pc,
                            max_length=self.max_code_length,
                            do_sample=True,  # Enable sampling for diversity
                            temperature=0.8,  # Moderate temperature for diversity
                            top_p=0.9,      # Nucleus sampling
                            pad_token_id=self.model.tokenizer.pad_token_id
                        )
                        
                        # Clean up generated code
                        if isinstance(generated_code, list):
                            generated_code = generated_code[0] if generated_code else ""
                        
                        cleaned_code = self._clean_generated_code(generated_code)
                        
                        if cleaned_code and len(cleaned_code.strip()) > 0:
                            candidates.append(cleaned_code)
                        else:
                            logger.warning(f"Empty code generated for candidate {candidate_idx}")
                            candidates.append("")  # Add empty string to maintain count
                        
                        generation_time = time.time() - candidate_start_time
                        generation_times.append(generation_time)
                        
                        # Log progress for long generations
                        if candidate_idx % 5 == 0 and candidate_idx > 0:
                            avg_time = np.mean(generation_times)
                            logger.debug(f"Generated {candidate_idx + 1}/{num_candidates} candidates, avg time: {avg_time:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Error generating candidate {candidate_idx}: {e}")
                        candidates.append("")  # Add empty string for failed generation
                        generation_times.append(0.0)
                        
                        # Continue with next candidate
                        continue
            
            # Filter out empty candidates
            valid_candidates = [c for c in candidates if c.strip()]
            
            if len(valid_candidates) == 0:
                logger.warning("No valid candidates generated")
                return []
            
            # Log generation statistics
            avg_generation_time = np.mean(generation_times) if generation_times else 0.0
            logger.debug(f"Generated {len(valid_candidates)}/{num_candidates} valid candidates in {avg_generation_time:.2f}s avg")
            
            return valid_candidates
            
        except Exception as e:
            logger.error(f"Error in candidate generation: {e}")
            raise RuntimeError(f"Candidate generation failed: {e}")
    
    def _apply_candidate_sampling(
        self, 
        point_cloud: torch.Tensor, 
        candidate_idx: int, 
        num_candidates: int
    ) -> torch.Tensor:
        """
        Apply different sampling strategy for each candidate.
        
        Args:
            point_cloud: Original point cloud tensor
            candidate_idx: Current candidate index
            num_candidates: Total number of candidates
            
        Returns:
            Sampled point cloud tensor of shape (num_points, 3)
        """
        # Set different random seed for each candidate
        torch.manual_seed(42 + candidate_idx)
        np.random.seed(42 + candidate_idx)
        
        # Apply furthest point sampling
        sampled_pc = furthest_point_sampling(point_cloud, self.num_points)
        
        # Apply optional noise augmentation for diversity
        if self.enable_noise_augmentation and candidate_idx > 0:  # Skip noise for first candidate
            # Vary noise intensity across candidates
            noise_factor = (candidate_idx / num_candidates) * 0.5  # Scale noise by candidate index
            current_noise_std = self.noise_std * noise_factor
            
            sampled_pc = add_gaussian_noise(
                sampled_pc,
                noise_std=current_noise_std,
                probability=1.0  # Always apply noise when enabled
            )
        
        return sampled_pc
    
    def _clean_generated_code(self, code: str) -> str:
        """
        Clean and normalize generated CAD code.
        
        Args:
            code: Raw generated code string
            
        Returns:
            Cleaned code string
        """
        if not isinstance(code, str):
            return ""
        
        # Remove special tokens
        start_token = self.config.get('model', {}).get('start_token', '<s>')
        end_token = self.config.get('model', {}).get('end_token', '<e>')
        
        cleaned = code.strip()
        
        # Remove start/end tokens
        if cleaned.startswith(start_token):
            cleaned = cleaned[len(start_token):].strip()
        
        if cleaned.endswith(end_token):
            cleaned = cleaned[:-len(end_token)].strip()
        
        # Remove any remaining special tokens or artifacts
        cleaned = cleaned.replace('<pad>', '').replace('<unk>', '').strip()
        
        # Ensure code starts with import if it doesn't
        if cleaned and not cleaned.startswith('import'):
            library_import = self.config.get('cadquery', {}).get('library_import', 'import cadquery as cq')
            if library_import not in cleaned:
                cleaned = f"{library_import}\n{cleaned}"
        
        return cleaned
    
    def evaluate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance on a specific dataset.
        
        Args:
            dataset_name: Name of dataset to evaluate ('deepcad', 'fusion360', 'cc3d')
            
        Returns:
            Dictionary containing evaluation results and metrics
            
        Raises:
            ValueError: If dataset name is invalid
            RuntimeError: If evaluation fails
        """
        if dataset_name not in self.test_loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.test_loaders.keys())}")
        
        logger.info(f"Starting evaluation on {dataset_name} dataset...")
        dataset_start_time = time.time()
        
        # Get test data loader
        test_loader = self.test_loaders[dataset_name]
        
        # Initialize result tracking
        sample_results = []
        metric_values = {
            'chamfer_distances': [],
            'ious': [],
            'invalid_count': 0,
            'total_count': 0
        }
        
        # Progress tracking
        total_samples = len(test_loader)
        progress_bar = tqdm(
            test_loader, 
            desc=f"Evaluating {dataset_name}",
            total=total_samples,
            disable=not logger.isEnabledFor(logging.INFO)
        )
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()
                
                try:
                    # Extract sample data
                    sample_id = batch.get('sample_id', [f"{dataset_name}_{batch_idx}"])[0]
                    point_cloud = batch['point_cloud'][0]  # Remove batch dimension
                    gt