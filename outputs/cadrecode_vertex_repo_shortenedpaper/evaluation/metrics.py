```python
"""
Metrics calculator for CAD-Recode system.
Implements Chamfer Distance, Intersection over Union, and Invalidity Ratio metrics
for evaluating CAD reverse engineering performance.
"""

import gc
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import trimesh

from config import Config
from utils.cad_utils import CADUtils
from utils.point_cloud_utils import PointCloudUtils


class MetricsCalculator:
    """
    Calculator for CAD-Recode evaluation metrics.
    Implements Chamfer Distance, Intersection over Union, and Invalidity Ratio.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize metrics calculator.
        
        Args:
            config: Configuration object with evaluation parameters
        """
        self.config = config if config is not None else Config()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize utilities
        self.cad_utils = CADUtils(self.config)
        
        # Evaluation parameters from config
        self.voxel_resolution = getattr(self.config.evaluation, 'voxel_resolution', 64)
        self.batch_size = getattr(self.config.evaluation, 'batch_size', 16)
        self.num_workers = getattr(self.config.evaluation, 'num_workers', 4)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance optimization settings
        self.use_gpu_acceleration = torch.cuda.is_available()
        self.cache_voxelizations = True
        self.voxel_cache: Dict[str, torch.Tensor] = {}
        
        # Statistics tracking
        self.computation_stats = {
            'chamfer_distance_calls': 0,
            'iou_calls': 0,
            'invalidity_ratio_calls': 0,
            'total_computation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger.info(f"Initialized MetricsCalculator: voxel_resolution={self.voxel_resolution}, "
                        f"device={self.device}, gpu_acceleration={self.use_gpu_acceleration}")
    
    def chamfer_distance(self, pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> float:
        """
        Compute Chamfer Distance between predicted and ground truth point clouds.
        
        Args:
            pred_pc: Predicted point cloud with shape [N, 3]
            gt_pc: Ground truth point cloud with shape [M, 3]
            
        Returns:
            float: Chamfer distance (lower is better)
            
        Raises:
            ValueError: If input tensors have invalid shapes or contain invalid values
        """
        start_time = time.time()
        self.computation_stats['chamfer_distance_calls'] += 1
        
        try:
            # Validate inputs
            self._validate_point_cloud_inputs(pred_pc, gt_pc)
            
            # Handle edge cases
            if len(pred_pc) == 0 and len(gt_pc) == 0:
                return 0.0
            elif len(pred_pc) == 0 or len(gt_pc) == 0:
                # Return maximum distance for empty point clouds
                return float('inf')
            
            # Convert to appropriate device and dtype
            if self.use_gpu_acceleration and pred_pc.device.type == 'cpu':
                pred_pc = pred_pc.to(self.device)
                gt_pc = gt_pc.to(self.device)
            
            # Ensure float32 for numerical stability
            pred_pc = pred_pc.float()
            gt_pc = gt_pc.float()
            
            # Compute Chamfer distance using efficient implementation
            if self.use_gpu_acceleration and pred_pc.is_cuda:
                cd = self._chamfer_distance_gpu(pred_pc, gt_pc)
            else:
                cd = self._chamfer_distance_cpu(pred_pc, gt_pc)
            
            # Convert to Python float
            if isinstance(cd, torch.Tensor):
                cd = cd.item()
            
            # Update statistics
            computation_time = time.time() - start_time
            self.computation_stats['total_computation_time'] += computation_time
            
            self.logger.debug(f"Chamfer distance computed: {cd:.6f} (time: {computation_time:.3f}s)")
            
            return float(cd)
            
        except Exception as e:
            self.logger.error(f"Chamfer distance computation failed: {e}")
            return float('inf')
    
    def _validate_point_cloud_inputs(self, pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> None:
        """
        Validate point cloud inputs for metric computation.
        
        Args:
            pred_pc: Predicted point cloud
            gt_pc: Ground truth point cloud
            
        Raises:
            ValueError: If inputs are invalid
        """
        for name, pc in [('pred_pc', pred_pc), ('gt_pc', gt_pc)]:
            if not isinstance(pc, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            
            if pc.ndim != 2 or pc.shape[1] != 3:
                raise ValueError(f"{name} must have shape [N, 3], got {pc.shape}")
            
            if len(pc) > 0 and not torch.isfinite(pc).all():
                raise ValueError(f"{name} contains invalid coordinates (NaN/Inf)")
    
    def _chamfer_distance_gpu(self, pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated Chamfer distance computation.
        
        Args:
            pred_pc: Predicted point cloud [N, 3]
            gt_pc: Ground truth point cloud [M, 3]
            
        Returns:
            torch.Tensor: Chamfer distance
        """
        # Compute pairwise distances efficiently
        # pred_pc: [N, 3], gt_pc: [M, 3]
        # distances: [N, M]
        
        # Use broadcasting to compute all pairwise distances
        pred_expanded = pred_pc.unsqueeze(1)  # [N, 1, 3]
        gt_expanded = gt_pc.unsqueeze(0)      # [1, M, 3]
        
        # Compute squared distances for numerical stability
        diff = pred_expanded - gt_expanded     # [N, M, 3]
        distances_squared = torch.sum(diff ** 2, dim=2)  # [N, M]
        
        # Find minimum distances in both directions
        pred_to_gt_min, _ = torch.min(distances_squared, dim=1)  # [N]
        gt_to_pred_min, _ = torch.min(distances_squared, dim=0)  # [M]
        
        # Compute Chamfer distance (mean of both directions)
        chamfer_dist = (torch.mean(pred_to_gt_min) + torch.mean(gt_to_pred_min)) / 2.0
        
        return torch.sqrt(chamfer_dist)  # Take square root for actual distance
    
    def _chamfer_distance_cpu(self, pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> float:
        """
        CPU-based Chamfer distance computation using scipy.
        
        Args:
            pred_pc: Predicted point cloud [N, 3]
            gt_pc: Ground truth point cloud [M, 3]
            
        Returns:
            float: Chamfer distance
        """
        # Convert to numpy for scipy operations
        pred_np = pred_pc.detach().cpu().numpy()
        gt_np = gt_pc.detach().cpu().numpy()
        
        # Build KD-trees for efficient nearest neighbor search
        try:
            pred_tree = cKDTree(pred_np)
            gt_tree = cKDTree(gt_np)
            
            # Find nearest neighbors in both directions
            pred_to_gt_distances, _ = pred_tree.query(gt_np, k=1)
            gt_to_pred_distances, _ = gt_tree.query(pred_np, k=1)
            
            # Compute Chamfer distance
            chamfer_dist = (np.mean(pred_to_gt_distances) + np.mean(gt_to_pred_distances)) / 2.0
            
            return float(chamfer_dist)
            
        except Exception as e:
            self.logger.warning(f"KD-tree approach failed, using fallback: {e}")
            # Fallback to direct distance computation
            return self._chamfer_distance_fallback(pred_np, gt_np)
    
    def _chamfer_distance_fallback(self, pred_np: np.ndarray, gt_np: np.ndarray) -> float:
        """
        Fallback Chamfer distance computation using direct distance matrix.
        
        Args:
            pred_np: Predicted point cloud [N, 3]
            gt_np: Ground truth point cloud [M, 3]
            
        Returns:
            float: Chamfer distance
        """
        try:
            # Compute distance matrices
            pred_to_gt_distances = cdist(pred_np, gt_np, metric='euclidean')
            gt_to_pred_distances = cdist(gt_np, pred_np, metric='euclidean')
            
            # Find minimum distances
            pred_to_gt_min = np.min(pred_to_gt_distances, axis=1)
            gt_to_pred_min = np.min(gt_to_pred_distances, axis=1)
            
            # Compute Chamfer distance
            chamfer_dist = (np.mean(pred_to_gt_min) + np.mean(gt_to_pred_min)) / 2.0
            
            return float(chamfer_dist)
            
        except Exception as e:
            self.logger.error(f"Fallback Chamfer distance computation failed: {e}")
            return float('inf')
    
    def intersection_over_union(self, pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> float:
        """
        Compute 3D Intersection over Union using voxelization.
        
        Args:
            pred_pc: Predicted point cloud with shape [N, 3]
            gt_pc: Ground truth point cloud with shape [M, 3]
            
        Returns:
            float: IoU score between 0 and 1 (higher is better)
        """
        start_time = time.time()
        self.computation_stats['iou_calls'] += 1
        
        try:
            # Validate inputs
            self._validate_point_cloud_inputs(pred_pc, gt_pc)
            
            # Handle edge cases
            if len(pred_pc) == 0 and len(gt_pc) == 0:
                return 1.0  # Perfect match for empty point clouds
            elif len(pred_pc) == 0 or len(gt_pc) == 0:
                return 0.0  # No overlap with empty point cloud
            
            # Convert point clouds to voxel grids
            pred_voxels = self._point_cloud_to_voxels(pred_pc, "pred")
            gt_voxels = self._point_cloud_to_voxels(gt_pc, "gt")
            
            # Compute intersection and union
            intersection = torch.logical_and(pred_voxels, gt_voxels)
            union = torch.logical_or(pred_voxels, gt_voxels)
            
            # Calculate IoU
            intersection_volume = torch.sum(intersection).float()
            union_volume = torch.sum(union).float()
            
            if union_volume == 0:
                iou = 1.0  # Both are empty
            else:
                iou = (intersection_volume / union_volume).item()
            
            # Update statistics
            computation_time = time.time() - start_time
            self.computation_stats['total_computation_time'] += computation_time
            
            self.logger.debug(f"IoU computed: {iou:.6f} (time: {computation_time:.3f}s)")
            
            return float(iou)
            
        except Exception as e:
            self.logger.error(f"IoU computation failed: {e}")
            return 0.0
    
    def _point_cloud_to_voxels(self, pc: torch.Tensor, cache_key: str = "") -> torch.Tensor:
        """
        Convert point cloud to binary voxel grid.
        
        Args:
            pc: Point cloud with shape [N, 3]
            cache_key: Optional cache key for voxelization caching
            
        Returns:
            torch.Tensor: Binary voxel grid [res, res, res]
        """
        # Check cache if key provided
        if cache_key and self.cache_voxelizations:
            cache_hash = f"{cache_key}_{hash(pc.data_ptr())}_{self.voxel_resolution}"
            if cache_hash in self.voxel_cache:
                self.computation_stats['cache_hits'] += 1
                return self.voxel_cache[cache_hash]
            else:
                self.computation_stats['cache_misses'] += 1
        
        try:
            # Normalize point cloud to unit cube [0, 1]^3
            pc_normalized = self._normalize_to_unit_cube(pc)
            
            # Convert to voxel coordinates
            voxel_coords = (pc_normalized * (self.voxel_resolution - 1)).long()
            
            # Clamp to valid range
            voxel_coords = torch.clamp(voxel_coords, 0, self.voxel_resolution - 1)
            
            # Create binary voxel grid
            voxel_grid = torch.zeros(
                (self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
                dtype=torch.bool,
                device=pc.device
            )
            
            # Set occupied voxels
            if len(voxel_coords) > 0:
                voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True
            
            # Cache result if key provided
            if cache_key and self.cache_voxelizations:
                self.voxel_cache[cache_hash] = voxel_grid
                
                # Limit cache size to prevent memory issues
                if len(self.voxel_cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = list(self.voxel_cache.keys())[:100]
                    for key in oldest_keys:
                        del self.voxel_cache[key]
            
            return voxel_grid
            
        except Exception as e:
            self.logger.error(f"Voxelization failed: {e}")
            # Return empty voxel grid as fallback
            return torch.zeros(
                (self.voxel_resolution, self.voxel_resolution, self.voxel_resolution),
                dtype=torch.bool,
                device=pc.device
            )
    
    def _normalize_to_unit_cube(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Normalize point cloud to unit cube [0, 1]^3.
        
        Args:
            pc: Point cloud with shape [N, 3]
            
        Returns:
            torch.Tensor: Normalized point cloud
        """
        if len(pc) == 0:
            return pc
        
        #