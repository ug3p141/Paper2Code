```python
"""
Evaluator module for CAD-Recode system.
Provides comprehensive evaluation across multiple datasets with Chamfer Distance, IoU, and Invalidity Ratio metrics.
"""

import gc
import json
import logging
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from models.cad_recode_model import CADRecodeModel
from evaluation.metrics import MetricsCalculator
from data.dataset_loader import CADDataset
from utils.cad_utils import CADUtils
from utils.point_cloud_utils import PointCloudUtils


class Evaluator:
    """
    Comprehensive evaluator for CAD-Recode model.
    Evaluates model performance on DeepCAD, Fusion360, and CC3D datasets using
    Chamfer Distance, Intersection over Union, and Invalidity Ratio metrics.
    """
    
    def __init__(self, 
                 model: CADRecodeModel,
                 config: Optional[Config] = None):
        """
        Initialize evaluator with model and configuration.
        
        Args:
            model: Trained CADRecodeModel for evaluation
            config: Configuration object with evaluation parameters
        """
        self.model = model
        self.config = config if config is not None else Config()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Device configuration
        self.device = torch.device(getattr(self.config.training, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Move model to evaluation device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize components
        self.metrics_calculator = MetricsCalculator(self.config)
        self.cad_utils = CADUtils(self.config)
        
        # Evaluation parameters from config
        self.batch_size = getattr(self.config.evaluation, 'batch_size', 16)
        self.num_workers = getattr(self.config.evaluation, 'num_workers', 4)
        self.metrics_list = getattr(self.config.evaluation, 'metrics', 
                                   ['chamfer_distance', 'intersection_over_union', 'invalidity_ratio'])
        self.results_dir = Path(getattr(self.config.evaluation, 'results_dir', './results'))
        self.visualizations_dir = Path(getattr(self.config.evaluation, 'visualizations_dir', './visualizations'))
        
        # Create output directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation parameters
        self.max_length = getattr(self.config.model.generation, 'max_length', 512)
        self.temperature = getattr(self.config.model.generation, 'temperature', 0.7)
        self.do_sample = getattr(self.config.model.generation, 'do_sample', True)
        self.top_p = getattr(self.config.model.generation, 'top_p', 0.9)
        
        # Dataset loaders storage
        self.dataset_loaders: Dict[str, DataLoader] = {}
        
        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}
        self.detailed_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.evaluation_stats = {
            'total_samples_processed': 0,
            'total_evaluation_time': 0.0,
            'average_inference_time': 0.0,
            'memory_usage_peak': 0.0,
            'dataset_processing_times': {}
        }
        
        self.logger.info(f"Initialized Evaluator: device={self.device}, "
                        f"batch_size={self.batch_size}, metrics={self.metrics_list}")
    
    def _setup_dataset_loaders(self) -> None:
        """Setup dataset loaders for all test datasets."""
        # Dataset configurations from config
        datasets_config = {
            'deepcad': {
                'path': getattr(self.config.dataset.test.deepcad, 'path', './data/deepcad'),
                'num_samples': getattr(self.config.dataset.test.deepcad, 'num_samples', 8046)
            },
            'fusion360': {
                'path': getattr(self.config.dataset.test.fusion360, 'path', './data/fusion360'),
                'num_samples': getattr(self.config.dataset.test.fusion360, 'num_samples', 1725)
            },
            'cc3d': {
                'path': getattr(self.config.dataset.test.cc3d, 'path', './data/cc3d'),
                'num_samples': getattr(self.config.dataset.test.cc3d, 'num_samples', 2973)
            }
        }
        
        for dataset_name, dataset_config in datasets_config.items():
            dataset_path = Path(dataset_config['path'])
            
            if not dataset_path.exists():
                self.logger.warning(f"Dataset path not found: {dataset_path}, skipping {dataset_name}")
                continue
            
            try:
                # Create dataset
                dataset = CADDataset(
                    data_dir=str(dataset_path),
                    tokenizer=self.model.tokenizer,
                    config=self.config,
                    split='test',
                    max_samples=dataset_config.get('num_samples')
                )
                
                # Create data loader
                data_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,  # Deterministic evaluation order
                    num_workers=self.num_workers,
                    pin_memory=True if self.device.type == 'cuda' else False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
                )
                
                self.dataset_loaders[dataset_name] = data_loader
                self.logger.info(f"Setup dataset loader for {dataset_name}: {len(dataset)} samples")
                
            except Exception as e:
                self.logger.error(f"Failed to setup dataset loader for {dataset_name}: {e}")
                continue
        
        if not self.dataset_loaders:
            raise ValueError("No valid dataset loaders created. Check dataset paths in config.")
    
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all configured test datasets.
        
        Returns:
            Dict[str, Dict[str, float]]: Evaluation results for each dataset
        """
        self.logger.info("Starting comprehensive evaluation")
        evaluation_start_time = time.time()
        
        # Setup dataset loaders
        self._setup_dataset_loaders()
        
        # Evaluate each dataset
        for dataset_name in self.dataset_loaders.keys():
            self.logger.info(f"Evaluating on {dataset_name} dataset")
            
            try:
                dataset_start_time = time.time()
                dataset_results = self._evaluate_dataset(dataset_name)
                dataset_time = time.time() - dataset_start_time
                
                self.results[dataset_name] = dataset_results
                self.evaluation_stats['dataset_processing_times'][dataset_name] = dataset_time
                
                self.logger.info(f"Completed {dataset_name} evaluation in {dataset_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {dataset_name}: {e}")
                self.logger.error(traceback.format_exc())
                continue
        
        # Calculate total evaluation time
        total_time = time.time() - evaluation_start_time
        self.evaluation_stats['total_evaluation_time'] = total_time
        
        # Generate evaluation report
        self._generate_evaluation_report()
        
        # Log summary
        self._log_evaluation_summary()
        
        self.logger.info(f"Evaluation completed in {total_time:.2f}s")
        
        return self.results
    
    def _evaluate_dataset(self, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate model on a single dataset.
        
        Args:
            dataset_name: Name of the dataset to evaluate
            
        Returns:
            Dict[str, float]: Evaluation metrics for the dataset
        """
        data_loader = self.dataset_loaders[dataset_name]
        
        # Initialize metric accumulators
        chamfer_distances = []
        ious = []
        invalid_codes = 0
        total_samples = 0
        inference_times = []
        
        # Initialize detailed results storage
        self.detailed_results[dataset_name] = []
        
        # Progress bar
        pbar = tqdm(data_loader, desc=f"Evaluating {dataset_name}")
        
        # Process batches
        for batch_idx, batch_data in enumerate(pbar):
            try:
                batch_results = self._evaluate_batch(batch_data, dataset_name, batch_idx)
                
                # Accumulate results
                if batch_results['chamfer_distances']:
                    chamfer_distances.extend(batch_results['chamfer_distances'])
                if batch_results['ious']:
                    ious.extend(batch_results['ious'])
                
                invalid_codes += batch_results['invalid_count']
                total_samples += batch_results['batch_size']
                
                if batch_results['inference_times']:
                    inference_times.extend(batch_results['inference_times'])
                
                # Update progress bar
                current_cd_mean = np.mean(chamfer_distances) if chamfer_distances else 0.0
                current_iou_mean = np.mean(ious) if ious else 0.0
                current_ir = invalid_codes / total_samples if total_samples > 0 else 0.0
                
                pbar.set_postfix({
                    'CD': f"{current_cd_mean:.4f}",
                    'IoU': f"{current_iou_mean:.4f}",
                    'IR': f"{current_ir:.3f}",
                    'Valid': f"{total_samples - invalid_codes}/{total_samples}"
                })
                
                # Memory cleanup
                if batch_idx % 10 == 0:  # Every 10 batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx} of {dataset_name}: {e}")
                continue
        
        # Compute final statistics
        return self._compute_dataset_statistics(
            chamfer_distances, ious, invalid_codes, total_samples, inference_times
        )
    
    def _evaluate_batch(self, 
                       batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                       dataset_name: str, 
                       batch_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single batch of data.
        
        Args:
            batch_data: Tuple of (point_clouds, input_ids, attention_mask)
            dataset_name: Name of the dataset being evaluated
            batch_idx: Index of the current batch
            
        Returns:
            Dict[str, Any]: Batch evaluation results
        """
        point_clouds, input_ids, attention_mask = batch_data
        batch_size = point_clouds.shape[0]
        
        # Move to device
        point_clouds = point_clouds.to(self.device, non_blocking=True)
        
        # Initialize batch results
        batch_results = {
            'chamfer_distances': [],
            'ious': [],
            'invalid_count': 0,
            'batch_size': batch_size,
            'inference_times': [],
            'sample_details': []
        }
        
        # Process each sample in the batch
        for i in range(batch_size):
            sample_start_time = time.time()
            
            try:
                # Generate CAD code for single sample
                sample_pc = point_clouds[i:i+1]  # Keep batch dimension
                
                inference_start = time.time()
                predicted_code = self.model.generate(
                    sample_pc,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    top_p=self.top_p
                )
                inference_time = time.time() - inference_start
                
                batch_results['inference_times'].append(inference_time)
                
                # Validate and execute predicted code
                sample_results = self._evaluate_single_sample(
                    point_cloud=point_clouds[i],
                    predicted_code=predicted_code,
                    sample_idx=batch_idx * batch_size + i,
                    dataset_name=dataset_name
                )
                
                # Accumulate results
                if sample_results['valid']:
                    batch_results['chamfer_distances'].append(sample_results['chamfer_distance'])
                    batch_results['ious'].append(sample_results['iou'])
                else:
                    batch_results['invalid_count'] += 1
                
                # Store detailed results
                sample_details = {
                    'sample_idx': batch_idx * batch_size + i,
                    'inference_time': inference_time,
                    'predicted_code': predicted_code,
                    'valid': sample_results['valid'],
                    'error_message': sample_results.get('error_message', ''),
                    'chamfer_distance': sample_results.get('chamfer_distance', float('inf')),
                    'iou': sample_results.get('iou', 0.0)
                }
                
                batch_results['sample_details'].append(sample_details)
                self.detailed_results[dataset_name].append(sample_details)
                
                # Update statistics
                total_time = time.time() - sample_start_time
                self.evaluation_stats['total_samples_processed'] += 1
                
            except Exception as e:
                self.logger.debug(f"Sample {i} in batch {batch_idx} failed: {e}")
                batch_results['invalid_count'] += 1
                
                # Store failed sample details
                sample_details = {
                    'sample_idx': batch_idx * batch_size + i,
                    'inference_time': 0.0,
                    'predicted_code': '',
                    'valid': False,
                    'error_message': str(e),
                    'chamfer_distance': float('inf'),
                    'iou': 0.0
                }
                
                batch_results['sample_details'].append(sample_details)
                self.detailed_results[dataset_name].append(sample_details)
        
        return batch_results
    
    def _evaluate_single_sample(self, 
                               point_cloud: torch.Tensor,
                               predicted_code: str,
                               sample_idx: int,
                               dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate a single sample with predicted CAD code.
        
        Args:
            point_cloud: Ground truth point cloud [num_points, 3]
            predicted_code: Generated CAD code string
            sample_idx: Index of the sample
            dataset_name: Name of the dataset
            
        Returns:
            Dict[str, Any]: Sample evaluation results
        """
        try:
            # Validate CAD code
            if not self.cad_utils.validate_cad_code(predicted_code):
                return {
                    'valid': False,
                    'error_message': 'Invalid CAD code syntax'
                }
            
            # Execute CAD code
            