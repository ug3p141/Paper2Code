```python
"""
Point cloud processing utilities for CAD-Recode system.
Provides core functionality for point cloud preprocessing, sampling, and I/O operations.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
import torch
import open3d as o3d
from config import Config


class PointCloudUtils:
    """
    Utility class providing core point cloud processing functionality.
    Implements furthest point sampling, normalization, data augmentation, and I/O operations.
    """
    
    @staticmethod
    def load_point_cloud(path: str) -> torch.Tensor:
        """
        Load point cloud from various file formats.
        
        Args:
            path: Path to point cloud file
            
        Returns:
            torch.Tensor: Point cloud with shape [N, 3]
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or data is invalid
            RuntimeError: If file loading fails
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")
        
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext in ['.ply', '.pcd']:
                # Use Open3D for standard point cloud formats
                pcd = o3d.io.read_point_cloud(str(file_path))
                if len(pcd.points) == 0:
                    raise ValueError(f"Empty point cloud loaded from {path}")
                points = np.asarray(pcd.points, dtype=np.float32)
                
            elif file_ext in ['.xyz', '.txt']:
                # Load ASCII format point clouds
                try:
                    points = np.loadtxt(str(file_path), dtype=np.float32)
                except ValueError as e:
                    # Try comma-separated format
                    points = np.loadtxt(str(file_path), delimiter=',', dtype=np.float32)
                
                if points.ndim == 1:
                    points = points.reshape(-1, 3)
                    
            elif file_ext == '.npy':
                # Load NumPy array format
                points = np.load(str(file_path)).astype(np.float32)
                if points.ndim == 1:
                    points = points.reshape(-1, 3)
                    
            elif file_ext == '.npz':
                # Load compressed NumPy format
                data = np.load(str(file_path))
                if 'points' in data:
                    points = data['points'].astype(np.float32)
                elif 'point_cloud' in data:
                    points = data['point_cloud'].astype(np.float32)
                else:
                    # Use first array if no standard key found
                    key = list(data.keys())[0]
                    points = data[key].astype(np.float32)
                    
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Validate point cloud structure
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Invalid point cloud shape: {points.shape}. Expected [N, 3]")
            
            if len(points) == 0:
                raise ValueError(f"Empty point cloud loaded from {path}")
            
            # Check for invalid coordinates
            if not np.isfinite(points).all():
                warnings.warn(f"Point cloud contains invalid coordinates (NaN/Inf): {path}")
                # Remove invalid points
                valid_mask = np.isfinite(points).all(axis=1)
                points = points[valid_mask]
                if len(points) == 0:
                    raise ValueError(f"No valid points remaining after filtering: {path}")
            
            return torch.from_numpy(points)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load point cloud from {path}: {str(e)}")
    
    @staticmethod
    def save_point_cloud(pc: torch.Tensor, path: str) -> None:
        """
        Save point cloud to file in specified format.
        
        Args:
            pc: Point cloud tensor with shape [N, 3]
            path: Output file path
            
        Raises:
            ValueError: If point cloud has invalid shape
            RuntimeError: If file saving fails
        """
        if not isinstance(pc, torch.Tensor):
            raise ValueError("Point cloud must be a torch.Tensor")
        
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape: {pc.shape}. Expected [N, 3]")
        
        if len(pc) == 0:
            raise ValueError("Cannot save empty point cloud")
        
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_ext = file_path.suffix.lower()
        
        # Convert to numpy for saving
        points_np = pc.detach().cpu().numpy().astype(np.float32)
        
        try:
            if file_ext in ['.ply', '.pcd']:
                # Use Open3D for standard formats
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_np)
                o3d.io.write_point_cloud(str(file_path), pcd)
                
            elif file_ext in ['.xyz', '.txt']:
                # Save as ASCII format
                np.savetxt(str(file_path), points_np, fmt='%.6f', delimiter=' ')
                
            elif file_ext == '.npy':
                # Save as NumPy array
                np.save(str(file_path), points_np)
                
            elif file_ext == '.npz':
                # Save as compressed NumPy format
                np.savez_compressed(str(file_path), points=points_np)
                
            else:
                raise ValueError(f"Unsupported output format: {file_ext}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to save point cloud to {path}: {str(e)}")
    
    @staticmethod
    def furthest_point_sampling(points: torch.Tensor, num_samples: int = 256) -> torch.Tensor:
        """
        Perform furthest point sampling to downsample point cloud.
        
        Args:
            points: Input point cloud with shape [N, 3]
            num_samples: Number of points to sample (default: 256 from config)
            
        Returns:
            torch.Tensor: Sampled point cloud with shape [num_samples, 3]
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(points, torch.Tensor):
            raise ValueError("Points must be a torch.Tensor")
        
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Invalid points shape: {points.shape}. Expected [N, 3]")
        
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        N = points.shape[0]
        if N == 0:
            raise ValueError("Cannot sample from empty point cloud")
        
        # If we have fewer points than requested, return all points with padding/repetition
        if N <= num_samples:
            if N == num_samples:
                return points.clone()
            else:
                # Repeat points to reach desired number
                indices = torch.arange(N, device=points.device)
                repeated_indices = indices.repeat((num_samples // N) + 1)[:num_samples]
                return points[repeated_indices]
        
        device = points.device
        dtype = points.dtype
        
        # Initialize arrays
        sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
        distances = torch.full((N,), float('inf'), dtype=dtype, device=device)
        
        # Start with a random point to ensure diversity
        torch.manual_seed(42)  # For reproducibility
        first_idx = torch.randint(0, N, (1,), device=device)
        sampled_indices[0] = first_idx
        
        # Update distances to first point
        first_point = points[first_idx]  # Shape: [1, 3]
        dists_to_first = torch.norm(points - first_point, dim=1)  # Shape: [N]
        distances = torch.minimum(distances, dists_to_first)
        
        # Iteratively select furthest points
        for i in range(1, num_samples):
            # Find point with maximum distance to any selected point
            furthest_idx = torch.argmax(distances)
            sampled_indices[i] = furthest_idx
            
            # Update distances with new point
            new_point = points[furthest_idx]  # Shape: [3]
            dists_to_new = torch.norm(points - new_point.unsqueeze(0), dim=1)  # Shape: [N]
            distances = torch.minimum(distances, dists_to_new)
        
        return points[sampled_indices]
    
    @staticmethod
    def normalize_point_cloud(pc: torch.Tensor) -> torch.Tensor:
        """
        Normalize point cloud to unit scale centered at origin.
        
        Args:
            pc: Point cloud tensor with shape [N, 3]
            
        Returns:
            torch.Tensor: Normalized point cloud with shape [N, 3]
            
        Raises:
            ValueError: If point cloud has invalid shape
        """
        if not isinstance(pc, torch.Tensor):
            raise ValueError("Point cloud must be a torch.Tensor")
        
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape: {pc.shape}. Expected [N, 3]")
        
        if len(pc) == 0:
            return pc.clone()
        
        # Clone to avoid modifying original
        normalized_pc = pc.clone()
        
        # Center at origin
        centroid = torch.mean(normalized_pc, dim=0)  # Shape: [3]
        normalized_pc = normalized_pc - centroid.unsqueeze(0)  # Shape: [N, 3]
        
        # Scale to unit cube
        # Calculate bounding box
        min_coords = torch.min(normalized_pc, dim=0)[0]  # Shape: [3]
        max_coords = torch.max(normalized_pc, dim=0)[0]  # Shape: [3]
        
        # Get maximum dimension for uniform scaling
        bbox_size = max_coords - min_coords  # Shape: [3]
        max_dim = torch.max(bbox_size)
        
        # Handle degenerate cases (single point or collinear points)
        if max_dim > 1e-8:  # Avoid division by very small numbers
            normalized_pc = normalized_pc / max_dim
        else:
            # For degenerate cases, return centered points without scaling
            warnings.warn("Point cloud has very small bounding box, skipping scaling")
        
        return normalized_pc
    
    @staticmethod
    def add_noise(pc: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """
        Add Gaussian noise to point cloud for data augmentation.
        
        Args:
            pc: Point cloud tensor with shape [N, 3]
            noise_level: Standard deviation of noise relative to point cloud scale
            
        Returns:
            torch.Tensor: Noisy point cloud with shape [N, 3]
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(pc, torch.Tensor):
            raise ValueError("Point cloud must be a torch.Tensor")
        
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape: {pc.shape}. Expected [N, 3]")
        
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")
        
        if len(pc) == 0 or noise_level == 0:
            return pc.clone()
        
        # Calculate point cloud scale for relative noise
        bbox_min = torch.min(pc, dim=0)[0]
        bbox_max = torch.max(pc, dim=0)[0]
        bbox_size = bbox_max - bbox_min
        scale = torch.max(bbox_size)
        
        # Handle degenerate case
        if scale < 1e-8:
            scale = 1.0
        
        # Generate Gaussian noise
        noise = torch.randn_like(pc) * noise_level * scale
        
        return pc + noise
    
    @staticmethod
    def voxelize_point_cloud(pc: torch.Tensor, voxel_size: float = 0.01) -> torch.Tensor:
        """
        Voxelize point cloud by keeping one point per voxel.
        
        Args:
            pc: Point cloud tensor with shape [N, 3]
            voxel_size: Size of each voxel
            
        Returns:
            torch.Tensor: Voxelized point cloud
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(pc, torch.Tensor):
            raise ValueError("Point cloud must be a torch.Tensor")
        
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape: {pc.shape}. Expected [N, 3]")
        
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        
        if len(pc) == 0:
            return pc.clone()
        
        # Convert to voxel coordinates
        voxel_coords = torch.floor(pc / voxel_size).long()
        
        # Find unique voxels
        unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        
        # Keep first point in each voxel
        voxelized_points = []
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            indices = torch.where(mask)[0]
            voxelized_points.append(pc[indices[0]])
        
        return torch.stack(voxelized_points)
    
    @staticmethod
    def compute_point_cloud_bounds(pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bounding box of point cloud.
        
        Args:
            pc: Point cloud tensor with shape [N, 3]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (min_coords, max_coords) each with shape [3]
            
        Raises:
            ValueError: If point cloud has invalid shape
        """
        if not isinstance(pc, torch.Tensor):
            raise ValueError("Point cloud must be a torch.Tensor")
        
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"Invalid point cloud shape: {pc.shape}. Expected [N, 3]")
        
        if len(pc) == 0:
            # Return zero bounds for empty point cloud
            return torch.zeros(3), torch.zeros(3)
        
        min_coords = torch.min(pc, dim=0)[0]
        max_coords = torch.max(pc, dim=0)[0]
        
        return min_coords, max_coords
    
    @staticmethod
    def filter_outliers(pc: torch.Tensor, nb_neighbors: int = 20, std_ratio: float = 2.0) -> torch.Tensor:
        """
        Remove statistical outliers from point cloud.
        
        Args:
            pc: Point cloud tensor with shape [N, 3]
            nb_neighbors: Number of neighbors to consider
            std_ratio: Standard deviation ratio threshold
            
        Returns:
            torch.Tensor