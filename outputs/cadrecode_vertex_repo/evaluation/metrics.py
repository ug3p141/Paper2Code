```python
"""
Evaluation metrics for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the three core evaluation metrics from the paper:
1. Chamfer Distance (CD) - geometric accuracy between predicted and ground truth
2. Intersection over Union (IoU) - volumetric overlap between CAD models  
3. Invalidity Ratio (IR) - percentage of invalid generated CAD sequences

All metrics follow the exact specifications from Section 5.1 of the paper.
"""

import os
import sys
import time
import logging
import warnings
import tempfile
import subprocess
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import torch

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    warnings.warn("Trimesh not available. Mesh operations will be limited.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    warnings.warn("Open3D not available. Point cloud operations will be limited.")

try:
    import cadquery as cq
    from cadquery import exporters
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    warnings.warn("CadQuery not available. CAD execution will be limited.")

try:
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Using fallback distance computation.")

# Import project utilities
from utils.cad_validation import CADValidator


# Set up logging
logger = logging.getLogger(__name__)


class Metrics:
    """
    Evaluation metrics implementation for CAD-Recode.
    
    Implements the three core metrics from the paper:
    - Chamfer Distance (CD): Geometric accuracy using 8,192 points
    - Intersection over Union (IoU): Volumetric overlap from meshes
    - Invalidity Ratio (IR): Percentage of invalid CAD sequences
    
    All computations follow the exact specifications from Section 5.1.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics calculator with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        # Default configuration from config.yaml
        default_config = {
            'evaluation': {
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
            'cadquery': {
                'validation_timeout': 30
            },
            'system': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'num_workers': 4
            }
        }
        
        if config is None:
            config = default_config
        else:
            # Merge with defaults
            config = self._deep_merge(default_config, config)
        
        self.config = config
        
        # Extract metric configuration
        metrics_config = config.get('evaluation', {}).get('metrics', {})
        
        # Chamfer Distance settings
        cd_config = metrics_config.get('chamfer_distance', {})
        self.cd_num_points = cd_config.get('num_points', 8192)
        self.cd_scale_factor = cd_config.get('scale_factor', 1000)
        
        # IoU settings
        iou_config = metrics_config.get('intersection_over_union', {})
        self.iou_compute_from = iou_config.get('compute_from', 'meshes')
        
        # IR settings
        ir_config = metrics_config.get('invalidity_ratio', {})
        self.ir_check_syntax = ir_config.get('check_syntax', True)
        self.ir_check_cad_semantics = ir_config.get('check_cad_semantics', True)
        
        # System settings
        system_config = config.get('system', {})
        self.device = system_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = system_config.get('num_workers', 4)
        
        # CAD execution settings
        cadquery_config = config.get('cadquery', {})
        self.validation_timeout = cadquery_config.get('validation_timeout', 30)
        
        # Initialize validator
        self.validator = CADValidator(cadquery_config) if CADQUERY_AVAILABLE else None
        
        # Initialize computation cache
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            'cd_computations': 0,
            'iou_computations': 0,
            'ir_computations': 0,
            'execution_failures': 0,
            'mesh_failures': 0,
            'timeout_failures': 0
        }
        
        logger.info(f"Metrics initialized:")
        logger.info(f"  CD points: {self.cd_num_points}, scale: {self.cd_scale_factor}")
        logger.info(f"  IoU from: {self.iou_compute_from}")
        logger.info(f"  Validation timeout: {self.validation_timeout}s")
        logger.info(f"  Device: {self.device}")
    
    def compute_chamfer_distance(
        self, 
        pred_pc: np.ndarray, 
        gt_pc: np.ndarray
    ) -> float:
        """
        Compute Chamfer Distance between predicted and ground truth point clouds.
        
        Implements the CD metric from Section 5.1 with exact paper specifications:
        - Uses 8,192 points for computation
        - Applies scale factor of 1000 for reporting
        - Computes bidirectional Chamfer Distance
        
        Args:
            pred_pc: Predicted point cloud of shape (N, 3)
            gt_pc: Ground truth point cloud of shape (M, 3)
            
        Returns:
            Scaled Chamfer Distance (multiplied by 1000)
            
        Raises:
            ValueError: If input point clouds are invalid
            RuntimeError: If computation fails
        """
        # Input validation
        if not isinstance(pred_pc, np.ndarray) or not isinstance(gt_pc, np.ndarray):
            raise ValueError("Point clouds must be numpy arrays")
        
        if pred_pc.ndim != 2 or pred_pc.shape[1] != 3:
            raise ValueError(f"Expected pred_pc shape (N, 3), got {pred_pc.shape}")
        
        if gt_pc.ndim != 2 or gt_pc.shape[1] != 3:
            raise ValueError(f"Expected gt_pc shape (M, 3), got {gt_pc.shape}")
        
        if pred_pc.shape[0] == 0 or gt_pc.shape[0] == 0:
            raise ValueError("Point clouds cannot be empty")
        
        # Check for invalid values
        if np.isnan(pred_pc).any() or np.isinf(pred_pc).any():
            raise ValueError("Predicted point cloud contains invalid values")
        
        if np.isnan(gt_pc).any() or np.isinf(gt_pc).any():
            raise ValueError("Ground truth point cloud contains invalid values")
        
        try:
            self.stats['cd_computations'] += 1
            
            # Sample points to target number (8,192)
            pred_sampled = self._sample_points(pred_pc, self.cd_num_points)
            gt_sampled = self._sample_points(gt_pc, self.cd_num_points)
            
            # Compute Chamfer Distance
            if torch.cuda.is_available() and self.device == 'cuda':
                # Use GPU acceleration if available
                cd_value = self._compute_chamfer_distance_gpu(pred_sampled, gt_sampled)
            else:
                # Use CPU computation
                cd_value = self._compute_chamfer_distance_cpu(pred_sampled, gt_sampled)
            
            # Apply scale factor (multiply by 1000 as per paper)
            scaled_cd = cd_value * self.cd_scale_factor
            
            return float(scaled_cd)
            
        except Exception as e:
            logger.error(f"Error computing Chamfer Distance: {e}")
            raise RuntimeError(f"Chamfer Distance computation failed: {e}")
    
    def compute_iou(
        self, 
        pred_mesh: Any, 
        gt_mesh: Any
    ) -> float:
        """
        Compute Intersection over Union between predicted and ground truth meshes.
        
        Implements the IoU metric from Section 5.1:
        - Computed from resulting CAD model meshes
        - Expressed as percentage
        - Uses volumetric intersection and union
        
        Args:
            pred_mesh: Predicted mesh (trimesh.Trimesh or compatible)
            gt_mesh: Ground truth mesh (trimesh.Trimesh or compatible)
            
        Returns:
            IoU value as percentage (0-100)
            
        Raises:
            ValueError: If meshes are invalid
            RuntimeError: If computation fails
        """
        if not TRIMESH_AVAILABLE:
            raise RuntimeError("Trimesh is required for IoU computation")
        
        try:
            self.stats['iou_computations'] += 1
            
            # Convert to trimesh objects if needed
            pred_trimesh = self._ensure_trimesh(pred_mesh)
            gt_trimesh = self._ensure_trimesh(gt_mesh)
            
            # Validate meshes
            if not pred_trimesh.is_volume or not gt_trimesh.is_volume:
                logger.warning("One or both meshes are not watertight volumes")
                # Attempt to fix meshes
                pred_trimesh = self._fix_mesh(pred_trimesh)
                gt_trimesh = self._fix_mesh(gt_trimesh)
            
            # Compute volumes
            pred_volume = self._compute_mesh_volume(pred_trimesh)
            gt_volume = self._compute_mesh_volume(gt_trimesh)
            
            if pred_volume <= 0 or gt_volume <= 0:
                logger.warning("One or both meshes have zero or negative volume")
                return 0.0
            
            # Compute intersection volume
            intersection_volume = self._compute_mesh_intersection(pred_trimesh, gt_trimesh)
            
            # Compute union volume
            union_volume = pred_volume + gt_volume - intersection_volume
            
            if union_volume <= 0:
                logger.warning("Union volume is zero or negative")
                return 0.0
            
            # Compute IoU as percentage
            iou = (intersection_volume / union_volume) * 100.0
            
            # Clamp to valid range
            iou = max(0.0, min(100.0, iou))
            
            return float(iou)
            
        except Exception as e:
            logger.error(f"Error computing IoU: {e}")
            self.stats['mesh_failures'] += 1
            # Return 0 for failed computations to avoid breaking evaluation
            return 0.0
    
    def compute_invalidity_ratio(self, codes: List[str]) -> float:
        """
        Compute Invalidity Ratio for a list of generated CAD codes.
        
        Implements the IR metric from Section 5.1:
        - Validates both syntax (φ_syn) and CAD semantics (φ_cad)
        - Returns percentage of invalid sequences
        
        Args:
            codes: List of CAD code strings to validate
            
        Returns:
            Invalidity ratio as percentage (0-100)
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(codes, list):
            raise ValueError("Codes must be a list of strings")
        
        if len(codes) == 0:
            return 0.0
        
        try:
            self.stats['ir_computations'] += 1
            
            invalid_count = 0
            total_count = len(codes)
            
            # Use parallel validation for efficiency
            if self.num_workers > 1 and total_count > 10:
                invalid_count = self._validate_codes_parallel(codes)
            else:
                invalid_count = self._validate_codes_sequential(codes)
            
            # Compute invalidity ratio as percentage
            invalidity_ratio = (invalid_count / total_count) * 100.0
            
            return float(invalidity_ratio)
            
        except Exception as e:
            logger.error(f"Error computing Invalidity Ratio: {e}")
            raise RuntimeError(f"Invalidity Ratio computation failed: {e}")
    
    def execute_cad_code(self, code: str) -> Optional[Any]:
        """
        Execute CAD code and return the resulting CAD model.
        
        Args:
            code: CAD code string to execute
            
        Returns:
            CAD model object if successful, None if failed
        """
        if not CADQUERY_AVAILABLE:
            logger.warning("CadQuery not available, cannot execute CAD code")
            return None
        
        if not isinstance(code, str) or not code.strip():
            return None
        
        try:
            # Create a safe execution environment
            safe_globals = {
                'cadquery': cq,
                'cq': cq,
                'math': __import__('math'),
                'numpy': __import__('numpy'),
                'np': __import__('numpy')
            }
            
            safe_locals = {}
            
            # Execute with timeout
            def execute_with_timeout():
                exec(code, safe_globals, safe_locals)
                # Look for the result variable (typically 'r')
                if 'r' in safe_locals:
                    return safe_locals['r']
                elif 'result' in safe_locals:
                    return safe_locals['result']
                else:
                    # Try to find any CadQuery object
                    for var_name, var_value in safe_locals.items():
                        if hasattr(var_value, 'val') and hasattr(var_value, 'vertices'):
                            return var_value
                    return None
            
            # Use timeout for execution
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("CAD code execution timed out")
            
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.validation_timeout)
            
            try:
                result = execute_with_timeout()
                signal.alarm(0)  # Cancel timeout
                return result
            except TimeoutError:
                logger.warning(f"CAD code execution timed out after {self.validation_timeout}s")
                self.stats['timeout_failures'] += 1
                return None
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                
        except Exception as e:
            logger.error(f"Error executing CAD code: {e}")
            self.stats['execution_failures'] += 1
            return None
    
    def cad_to_point_cloud(
        self, 
        cad_model: Any, 
        num_points: int = 8192
    ) -> np.ndarray:
        """
        Convert CAD model to point cloud by sampling from mesh surface.
        
        Args:
            cad_model: CAD model object (CadQuery Workplane)
            num_points: Number of points to sample
            
        Returns:
            Point cloud array of shape (num_points, 3)
            
        Raises: