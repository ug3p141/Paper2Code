```python
"""
Point cloud processing utilities for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module provides essential point cloud processing utilities including furthest point
sampling, Gaussian noise augmentation, normalization, and validation. These utilities
serve as foundational components shared across the point cloud projector, dataset loader,
and evaluation pipeline.
"""

import torch
import numpy as np
import open3d as o3d
import trimesh
import warnings
from typing import Union, Tuple, List, Optional, Any, Callable, Dict
from pathlib import Path
import logging


# Set up logging
logger = logging.getLogger(__name__)


def furthest_point_sampling(
    point_cloud: Union[torch.Tensor, np.ndarray], 
    num_points: int = 256
) -> torch.Tensor:
    """
    Downsample point cloud using furthest point sampling algorithm.
    
    As specified in Section 4.2 of the paper, this function downsamples input
    point clouds to exactly num_points using the furthest point sampling strategy.
    This ensures diverse spatial coverage while maintaining geometric structure.
    
    Args:
        point_cloud: Input point cloud of shape (N, 3) where N >= num_points
        num_points: Target number of points (default: 256 from config)
        
    Returns:
        Downsampled point cloud tensor of shape (num_points, 3)
        
    Raises:
        ValueError: If input shape is invalid or num_points is invalid
        RuntimeError: If input contains invalid values
    """
    # Input validation
    if isinstance(point_cloud, np.ndarray):
        point_cloud = torch.from_numpy(point_cloud).float()
    
    if not isinstance(point_cloud, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(point_cloud)}")
    
    if point_cloud.dim() != 2 or point_cloud.size(-1) != 3:
        raise ValueError(f"Expected shape (N, 3), got {point_cloud.shape}")
    
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    
    # Check for invalid values
    if torch.isnan(point_cloud).any() or torch.isinf(point_cloud).any():
        raise RuntimeError("Input point cloud contains NaN or infinite values")
    
    N = point_cloud.size(0)
    
    # Handle edge case where input has fewer points than requested
    if N <= num_points:
        if N < num_points:
            warnings.warn(f"Input has {N} points, less than requested {num_points}. "
                         f"Repeating points to reach target size.")
            # Repeat points to reach target size
            repeat_factor = (num_points + N - 1) // N  # Ceiling division
            repeated_pc = point_cloud.repeat(repeat_factor, 1)
            return repeated_pc[:num_points]
        else:
            return point_cloud.clone()
    
    device = point_cloud.device
    
    # Initialize arrays for selected points and distances
    selected_indices = torch.zeros(num_points, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    
    # Select first point randomly for better coverage
    first_idx = torch.randint(0, N, (1,), device=device).item()
    selected_indices[0] = first_idx
    
    # Update distances to first point
    first_point = point_cloud[first_idx].unsqueeze(0)  # Shape: (1, 3)
    dists_to_first = torch.sum((point_cloud - first_point) ** 2, dim=1)  # Shape: (N,)
    distances = torch.minimum(distances, dists_to_first)
    
    # Iteratively select furthest points
    for i in range(1, num_points):
        # Find point with maximum distance to all selected points
        furthest_idx = torch.argmax(distances).item()
        selected_indices[i] = furthest_idx
        
        # Update distances with new selected point
        new_point = point_cloud[furthest_idx].unsqueeze(0)  # Shape: (1, 3)
        dists_to_new = torch.sum((point_cloud - new_point) ** 2, dim=1)  # Shape: (N,)
        distances = torch.minimum(distances, dists_to_new)
    
    # Return selected points
    sampled_points = point_cloud[selected_indices]
    
    return sampled_points


def add_gaussian_noise(
    point_cloud: Union[torch.Tensor, np.ndarray],
    noise_std: float = 0.01,
    probability: float = 0.5
) -> torch.Tensor:
    """
    Apply Gaussian noise augmentation to point cloud.
    
    As described in Section 4.3 training strategy, this function applies
    Gaussian noise with mean zero and specified standard deviation with
    a given probability for data augmentation during training.
    
    Args:
        point_cloud: Input point cloud of shape (..., 3)
        noise_std: Standard deviation of Gaussian noise (default: 0.01 from config)
        probability: Probability of applying noise (default: 0.5 from config)
        
    Returns:
        Point cloud with optional noise applied, same shape as input
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Input validation
    if isinstance(point_cloud, np.ndarray):
        point_cloud = torch.from_numpy(point_cloud).float()
    
    if not isinstance(point_cloud, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(point_cloud)}")
    
    if point_cloud.size(-1) != 3:
        raise ValueError(f"Expected last dimension to be 3, got {point_cloud.size(-1)}")
    
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    
    if not 0 <= probability <= 1:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    
    # Clone to avoid modifying original
    augmented_pc = point_cloud.clone()
    
    # Apply noise with given probability
    if torch.rand(1).item() < probability:
        # Generate Gaussian noise with same shape as input
        noise = torch.randn_like(augmented_pc) * noise_std
        augmented_pc = augmented_pc + noise
    
    return augmented_pc


def normalize_point_cloud(
    point_cloud: Union[torch.Tensor, np.ndarray],
    method: str = "unit_box"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Normalize point cloud to fit within unit bounding box.
    
    As specified in Appendix B dataset generation, this function normalizes
    point clouds to ensure consistent scale and centering for training and evaluation.
    
    Args:
        point_cloud: Input point cloud of shape (N, 3)
        method: Normalization method (default: "unit_box" from config)
        
    Returns:
        Tuple of (normalized_point_cloud, transformation_params)
        transformation_params contains 'center' and 'scale' for potential inverse transform
        
    Raises:
        ValueError: If input shape or method is invalid
        RuntimeError: If normalization fails due to degenerate input
    """
    # Input validation
    if isinstance(point_cloud, np.ndarray):
        point_cloud = torch.from_numpy(point_cloud).float()
    
    if not isinstance(point_cloud, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(point_cloud)}")
    
    if point_cloud.dim() != 2 or point_cloud.size(-1) != 3:
        raise ValueError(f"Expected shape (N, 3), got {point_cloud.shape}")
    
    if method not in ["unit_box", "unit_sphere", "zero_mean"]:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if point_cloud.size(0) == 0:
        return point_cloud.clone(), {'center': torch.zeros(3), 'scale': torch.tensor(1.0)}
    
    device = point_cloud.device
    
    if method == "unit_box":
        # Compute bounding box
        min_coords = torch.min(point_cloud, dim=0)[0]  # Shape: (3,)
        max_coords = torch.max(point_cloud, dim=0)[0]  # Shape: (3,)
        
        # Calculate center and scale
        center = (min_coords + max_coords) / 2.0  # Shape: (3,)
        extent = max_coords - min_coords  # Shape: (3,)
        
        # Handle degenerate cases
        max_extent = torch.max(extent)
        if max_extent < 1e-8:
            warnings.warn("Point cloud has very small extent, normalization may be unstable")
            scale = torch.tensor(1.0, device=device)
        else:
            scale = max_extent
        
        # Apply normalization: (coords - center) / scale
        normalized_pc = (point_cloud - center.unsqueeze(0)) / scale
        
    elif method == "unit_sphere":
        # Center at origin
        center = torch.mean(point_cloud, dim=0)  # Shape: (3,)
        centered_pc = point_cloud - center.unsqueeze(0)
        
        # Scale to unit sphere
        distances = torch.norm(centered_pc, dim=1)  # Shape: (N,)
        max_distance = torch.max(distances)
        
        if max_distance < 1e-8:
            warnings.warn("All points are at the same location")
            scale = torch.tensor(1.0, device=device)
        else:
            scale = max_distance
        
        normalized_pc = centered_pc / scale
        
    elif method == "zero_mean":
        # Simply center at origin without scaling
        center = torch.mean(point_cloud, dim=0)  # Shape: (3,)
        normalized_pc = point_cloud - center.unsqueeze(0)
        scale = torch.tensor(1.0, device=device)
    
    # Store transformation parameters
    transformation_params = {
        'center': center,
        'scale': scale,
        'method': method
    }
    
    return normalized_pc, transformation_params


def mesh_to_point_cloud(
    mesh: Any,
    num_points: int = 8192,
    method: str = "uniform"
) -> torch.Tensor:
    """
    Convert CAD mesh to point cloud for evaluation metrics computation.
    
    This function supports conversion from various mesh formats (trimesh, open3d)
    to point clouds with specified number of points for consistent evaluation.
    
    Args:
        mesh: Input mesh object (trimesh.Trimesh, o3d.geometry.TriangleMesh, etc.)
        num_points: Target number of points (default: 8192 from config)
        method: Sampling method ("uniform" or "random")
        
    Returns:
        Point cloud tensor of shape (num_points, 3)
        
    Raises:
        ValueError: If mesh format is unsupported or parameters are invalid
        RuntimeError: If sampling fails
    """
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    
    try:
        # Handle different mesh types
        if isinstance(mesh, trimesh.Trimesh):
            # Use trimesh sampling
            if method == "uniform":
                points, _ = trimesh.sample.sample_surface(mesh, num_points)
            else:  # random
                points = mesh.sample(num_points)
            points = torch.from_numpy(points).float()
            
        elif hasattr(mesh, 'sample_points_uniformly'):  # Open3D mesh
            # Convert to Open3D format if needed
            if not isinstance(mesh, o3d.geometry.TriangleMesh):
                raise ValueError(f"Unsupported mesh type: {type(mesh)}")
            
            # Sample points uniformly
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            points = torch.from_numpy(np.asarray(pcd.points)).float()
            
        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            # Generic mesh with vertices and faces
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            
            # Create trimesh object for sampling
            trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
            points, _ = trimesh.sample.sample_surface(trimesh_obj, num_points)
            points = torch.from_numpy(points).float()
            
        else:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}")
        
        # Validate output
        if points.shape[0] != num_points or points.shape[1] != 3:
            raise RuntimeError(f"Sampling failed: expected ({num_points}, 3), got {points.shape}")
        
        return points
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert mesh to point cloud: {e}")


def validate_point_cloud(
    point_cloud: Union[torch.Tensor, np.ndarray],
    min_points: int = 1,
    max_points: Optional[int] = None,
    coord_range: Optional[Tuple[float, float]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate point cloud format and properties.
    
    Args:
        point_cloud: Input point cloud to validate
        min_points: Minimum required number of points
        max_points: Maximum allowed number of points (None for no limit)
        coord_range: Valid coordinate range as (min_val, max_val)
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Check basic type and format
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud)
        
        if not isinstance(point_cloud, torch.Tensor):
            errors.append(f"Expected torch.Tensor or np.ndarray, got {type(point_cloud)}")
            return False, errors
        
        # Check shape
        if point_cloud.dim() != 2:
            errors.append(f"Expected 2D tensor, got {point_cloud.dim()}D")
        elif point_cloud.size(-1) != 3:
            errors.append(f"Expected 3D coordinates, got {point_cloud.size(-1)}D")
        
        # Check number of points
        num_points = point_cloud.size(0) if point_cloud.dim() >= 1 else 0
        if num_points < min_points:
            errors.append(f"Too few points: {num_points} < {min_points}")
        
        if max_points is not None and num_points > max_points:
            errors.append(f"Too many points: {num_points} > {max_points}")
        
        # Check for invalid values
        if torch.isnan(point_cloud).any():
            errors.append("Point cloud contains NaN values")
        
        if torch.isinf(point_cloud).any():
            errors.append("Point cloud contains infinite values")
        
        # Check coordinate range if specified
        if coord_range is not None and point_cloud.numel() > 0:
            min_val, max_val = coord_range
            pc_min = torch.min(point_cloud).item()
            pc_max = torch.max(point_cloud).item()
            
            if pc_min < min_val or pc_max > max_val:
                errors.append(f"Coordinates out of range [{min_val}, {max_val}]: "
                             f"actual range [{pc_