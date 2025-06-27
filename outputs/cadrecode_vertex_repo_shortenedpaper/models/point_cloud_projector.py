"""
Point cloud projector module for CAD-Recode system.
Implements lightweight projection from 3D point clouds to LLM-compatible query tokens.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from utils.point_cloud_utils import PointCloudUtils


class FourierPositionalEncoder(nn.Module):
    """
    Fourier positional encoding for 3D coordinates.
    Encodes spatial coordinates using sinusoidal functions with multiple frequency bands.
    """
    
    def __init__(self, num_freqs: int = 10):
        """
        Initialize Fourier positional encoder.
        
        Args:
            num_freqs: Number of frequency bands for encoding
        """
        super().__init__()
        
        if num_freqs <= 0:
            raise ValueError("num_freqs must be positive")
        
        self.num_freqs = num_freqs
        
        # Pre-compute frequency bands: 2^0, 2^1, ..., 2^(num_freqs-1)
        # Shape: [num_freqs]
        freq_bands = torch.pow(2.0, torch.arange(num_freqs, dtype=torch.float32))
        
        # Register as buffer so it moves with the model to GPU/CPU
        self.register_buffer('freq_bands', freq_bands)
        
        # Output dimension: 3 coords * 2 (sin/cos) * num_freqs
        self.output_dim = 3 * 2 * num_freqs
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier positional encoding to 3D coordinates.
        
        Args:
            coords: 3D coordinates with shape [batch_size, num_points, 3]
            
        Returns:
            torch.Tensor: Encoded coordinates with shape [batch_size, num_points, output_dim]
            
        Raises:
            ValueError: If input tensor has invalid shape
        """
        if not isinstance(coords, torch.Tensor):
            raise ValueError("coords must be a torch.Tensor")
        
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError(f"coords must have shape [batch_size, num_points, 3], got {coords.shape}")
        
        batch_size, num_points, _ = coords.shape
        device = coords.device
        dtype = coords.dtype
        
        # Ensure frequency bands are on the same device
        freq_bands = self.freq_bands.to(device=device, dtype=dtype)
        
        # Reshape coordinates for broadcasting
        # coords: [batch_size, num_points, 3]
        # freq_bands: [num_freqs]
        # We want: [batch_size, num_points, 3, num_freqs]
        coords_expanded = coords.unsqueeze(-1)  # [batch_size, num_points, 3, 1]
        freq_bands_expanded = freq_bands.view(1, 1, 1, self.num_freqs)  # [1, 1, 1, num_freqs]
        
        # Compute scaled coordinates: coord * 2^i * π
        scaled_coords = coords_expanded * freq_bands_expanded * math.pi
        # Shape: [batch_size, num_points, 3, num_freqs]
        
        # Apply sin and cos encodings
        sin_encoding = torch.sin(scaled_coords)  # [batch_size, num_points, 3, num_freqs]
        cos_encoding = torch.cos(scaled_coords)  # [batch_size, num_points, 3, num_freqs]
        
        # Stack sin and cos encodings
        # Shape: [batch_size, num_points, 3, num_freqs, 2]
        encoding = torch.stack([sin_encoding, cos_encoding], dim=-1)
        
        # Reshape to final output format
        # [batch_size, num_points, 3 * num_freqs * 2]
        encoded_coords = encoding.view(batch_size, num_points, self.output_dim)
        
        return encoded_coords


class PointCloudProjector(nn.Module):
    """
    Point cloud projector that converts 3D point clouds to LLM-compatible query tokens.
    Combines furthest point sampling, Fourier positional encoding, and linear projection.
    """
    
    def __init__(self, 
                 num_points: int = 256,
                 hidden_dim: int = 768,
                 fourier_freqs: int = 10):
        """
        Initialize point cloud projector.
        
        Args:
            num_points: Number of points after furthest point sampling
            hidden_dim: Output dimension for LLM compatibility
            fourier_freqs: Number of Fourier frequency bands for positional encoding
        """
        super().__init__()
        
        # Validate parameters
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if fourier_freqs <= 0:
            raise ValueError("fourier_freqs must be positive")
        
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.fourier_freqs = fourier_freqs
        
        # Initialize Fourier positional encoder
        self.fourier_encoder = FourierPositionalEncoder(num_freqs=fourier_freqs)
        
        # Linear projection layer
        # Input dimension: 3 coordinates * 2 (sin/cos) * fourier_freqs
        input_dim = 3 * 2 * fourier_freqs
        self.linear_proj = nn.Linear(input_dim, hidden_dim)
        
        # Initialize linear layer weights
        self._init_weights()
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized PointCloudProjector: "
                        f"num_points={num_points}, hidden_dim={hidden_dim}, "
                        f"fourier_freqs={fourier_freqs}, input_dim={input_dim}")
    
    def _init_weights(self) -> None:
        """Initialize linear projection weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.linear_proj.bias is not None:
            nn.init.zeros_(self.linear_proj.bias)
    
    @classmethod
    def from_config(cls, config: Config) -> 'PointCloudProjector':
        """
        Create PointCloudProjector from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            PointCloudProjector: Configured projector instance
        """
        return cls(
            num_points=config.model.projector.num_points,
            hidden_dim=config.model.projector.hidden_dim,
            fourier_freqs=config.model.projector.fourier_freqs
        )
    
    def furthest_point_sampling(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply furthest point sampling to downsample point cloud.
        
        Args:
            points: Input point cloud with shape [batch_size, num_points, 3]
            
        Returns:
            torch.Tensor: Sampled point cloud with shape [batch_size, self.num_points, 3]
            
        Raises:
            ValueError: If input tensor has invalid shape
        """
        if not isinstance(points, torch.Tensor):
            raise ValueError("points must be a torch.Tensor")
        
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(f"points must have shape [batch_size, num_points, 3], got {points.shape}")
        
        batch_size, input_num_points, _ = points.shape
        
        if input_num_points == 0:
            raise ValueError("Cannot sample from empty point cloud")
        
        # Process each sample in the batch
        sampled_batch = []
        
        for i in range(batch_size):
            sample_points = points[i]  # Shape: [input_num_points, 3]
            
            # Apply FPS using utility function
            try:
                sampled_points = PointCloudUtils.furthest_point_sampling(
                    sample_points, self.num_points
                )
                sampled_batch.append(sampled_points)
            except Exception as e:
                self.logger.error(f"FPS failed for batch item {i}: {e}")
                # Fallback: use random sampling
                if input_num_points >= self.num_points:
                    indices = torch.randperm(input_num_points)[:self.num_points]
                    sampled_points = sample_points[indices]
                else:
                    # Repeat points to reach desired number
                    repeat_factor = (self.num_points // input_num_points) + 1
                    repeated_points = sample_points.repeat(repeat_factor, 1)
                    sampled_points = repeated_points[:self.num_points]
                
                sampled_batch.append(sampled_points)
        
        # Stack batch results
        result = torch.stack(sampled_batch, dim=0)
        
        # Verify output shape
        assert result.shape == (batch_size, self.num_points, 3), \
            f"Unexpected output shape: {result.shape}"
        
        return result
    
    def _normalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates to improve Fourier encoding stability.
        
        Args:
            coords: Coordinates with shape [batch_size, num_points, 3]
            
        Returns:
            torch.Tensor: Normalized coordinates with same shape
        """
        batch_size, num_points, _ = coords.shape
        
        # Normalize each sample in the batch independently
        normalized_batch = []
        
        for i in range(batch_size):
            sample_coords = coords[i]  # Shape: [num_points, 3]
            
            # Center at origin
            centroid = torch.mean(sample_coords, dim=0, keepdim=True)  # Shape: [1, 3]
            centered_coords = sample_coords - centroid
            
            # Scale to unit cube
            # Compute bounding box
            min_coords = torch.min(centered_coords, dim=0)[0]  # Shape: [3]
            max_coords = torch.max(centered_coords, dim=0)[0]  # Shape: [3]
            bbox_size = max_coords - min_coords  # Shape: [3]
            max_dim = torch.max(bbox_size)
            
            # Handle degenerate cases (very small or zero bounding box)
            if max_dim > 1e-8:
                normalized_coords = centered_coords / max_dim
            else:
                # Keep centered coordinates without scaling
                normalized_coords = centered_coords
                self.logger.debug(f"Small bounding box detected in batch item {i}, skipping scaling")
            
            normalized_batch.append(normalized_coords)
        
        return torch.stack(normalized_batch, dim=0)
    
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert point cloud to query tokens.
        
        Args:
            point_cloud: Input point cloud with shape [batch_size, variable_points, 3]
            
        Returns:
            torch.Tensor: Query tokens with shape [batch_size, num_points, hidden_dim]
            
        Raises:
            ValueError: If input tensor has invalid shape or contains invalid values
        """
        if not isinstance(point_cloud, torch.Tensor):
            raise ValueError("point_cloud must be a torch.Tensor")
        
        if point_cloud.ndim != 3 or point_cloud.shape[2] != 3:
            raise ValueError(f"point_cloud must have shape [batch_size, num_points, 3], got {point_cloud.shape}")
        
        batch_size, input_num_points, _ = point_cloud.shape
        
        if batch_size == 0:
            raise ValueError("Batch size cannot be zero")
        
        if input_num_points == 0:
            raise ValueError("Point cloud cannot be empty")
        
        # Check for invalid coordinates (NaN, Inf)
        if not torch.isfinite(point_cloud).all():
            self.logger.warning("Point cloud contains invalid coordinates (NaN/Inf)")
            # Replace invalid coordinates with zeros
            point_cloud = torch.where(torch.isfinite(point_cloud), point_cloud, 
                                    torch.zeros_like(point_cloud))
        
        # Step 1: Apply furthest point sampling
        try:
            sampled_points = self.furthest_point_sampling(point_cloud)
            # Shape: [batch_size, num_points, 3]
        except Exception as e:
            self.logger.error(f"Furthest point sampling failed: {e}")
            raise ValueError(f"Point cloud sampling failed: {e}")
        
        # Step 2: Normalize coordinates for stable Fourier encoding
        normalized_points = self._normalize_coordinates(sampled_points)
        # Shape: [batch_size, num_points, 3]
        
        # Step 3: Apply Fourier positional encoding
        try:
            encoded_coords = self.fourier_encoder(normalized_points)
            # Shape: [batch_size, num_points, 3 * 2 * fourier_freqs]
        except Exception as e:
            self.logger.error(f"Fourier encoding failed: {e}")
            raise ValueError(f"Coordinate encoding failed: {e}")
        
        # Step 4: Linear projection to query tokens
        try:
            query_tokens = self.linear_proj(encoded_coords)
            # Shape: [batch_size, num_points, hidden_dim]
        except Exception as e:
            self.logger.error(f"Linear projection failed: {e}")
            raise ValueError(f"Token projection failed: {e}")
        
        # Verify output shape
        expected_shape = (batch_size, self.num_points, self.hidden_dim)
        if query_tokens.shape != expected_shape:
            raise RuntimeError(f"Unexpected output shape: {query_tokens.shape}, expected: {expected_shape}")
        
        # Check for invalid outputs
        if not torch.isfinite(query_tokens).all():
            self.logger.error("Query tokens contain invalid values (NaN/Inf)")
            raise RuntimeError("Generated query tokens contain invalid values")
        
        return query_tokens
    
    def get_output_dim(self) -> int:
        """
        Get output dimension of query tokens.
        
        Returns:
            int: Hidden dimension of output tokens
        """
        return self.hidden_dim
    
    def get_num_output_tokens(self) -> int:
        """
        Get number of output tokens per point cloud.
        
        Returns:
            int: Number of query tokens generated per input
        """
        return self.num_points
    
    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return (f'num_points={self.num_points}, hidden_dim={self.hidden_dim}, '
                f'fourier_freqs={self.fourier_freqs}')
