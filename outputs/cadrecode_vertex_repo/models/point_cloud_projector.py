```python
"""
Point cloud projector module for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the point cloud projector component (Ψ_p) from the CAD-Recode
architecture as described in Section 4.2. The projector transforms raw 3D point clouds
into learnable query tokens through furthest point sampling, Fourier positional encoding,
and linear projection.

The projector implements the mapping: Ψ_p: R^(n×3) → R^(n_p×d_q)
where n_p = 256 (number of sampled points) and d_q = 1536 (embedding dimension).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
from typing import Optional, Tuple, Union, Dict, Any

# Import utilities
from utils.fourier_encoding import FourierEncoding
from utils.point_cloud_utils import furthest_point_sampling, validate_point_cloud


# Set up logging
logger = logging.getLogger(__name__)


class PointCloudProjector(nn.Module):
    """
    Point cloud projector implementing Ψ_p from CAD-Recode architecture.
    
    This lightweight, trainable module transforms variable-size point clouds into
    fixed-size query tokens suitable for LLM processing. The module consists of
    three sequential components as specified in Section 4.2:
    
    1. Furthest Point Sampling: Downsample to n_p = 256 points
    2. Fourier Positional Encoding: Encode 3D coordinates 
    3. Linear Projection: Map to embedding dimension d_q = 1536
    """
    
    def __init__(
        self,
        num_points: int = 256,
        embed_dim: int = 1536,
        num_freqs: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize point cloud projector with specified parameters.
        
        Args:
            num_points: Number of points after furthest point sampling (default: 256)
            embed_dim: Output embedding dimension for query tokens (default: 1536)
            num_freqs: Number of Fourier encoding frequencies (default: 64)
            config: Optional configuration dictionary from config.yaml
        
        Raises:
            ValueError: If parameters are invalid
        """
        super(PointCloudProjector, self).__init__()
        
        # Load configuration if provided
        if config is not None:
            model_config = config.get('model', {})
            num_points = model_config.get('num_points', num_points)
            embed_dim = model_config.get('embedding_dim', embed_dim)
            
            fourier_config = model_config.get('fourier_encoding', {})
            if num_freqs is None:
                num_freqs = fourier_config.get('num_freqs', 64)
        
        # Set default values if not provided
        if num_freqs is None:
            num_freqs = 64
        
        # Validate parameters
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_freqs <= 0:
            raise ValueError(f"num_freqs must be positive, got {num_freqs}")
        
        # Store hyperparameters
        self.num_points = num_points
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs
        
        # Initialize Fourier positional encoding
        # This transforms 3D coordinates to higher-dimensional representations
        self.fourier_encoder = FourierEncoding(num_freqs=num_freqs)
        
        # Calculate input dimension for linear projection
        # Fourier encoding outputs 3 * 2 * num_freqs dimensions
        fourier_output_dim = self.fourier_encoder.get_output_dim()
        
        # Initialize linear projection layer
        # Maps from Fourier-encoded coordinates to LLM embedding space
        self.linear_proj = nn.Linear(fourier_output_dim, embed_dim)
        
        # Initialize linear layer with reasonable values
        self._initialize_weights()
        
        logger.info(f"PointCloudProjector initialized:")
        logger.info(f"  num_points: {self.num_points}")
        logger.info(f"  embed_dim: {self.embed_dim}")
        logger.info(f"  num_freqs: {self.num_freqs}")
        logger.info(f"  fourier_output_dim: {fourier_output_dim}")
    
    def _initialize_weights(self) -> None:
        """Initialize linear projection weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.zeros_(self.linear_proj.bias)
    
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of point cloud projector.
        
        Transforms input point clouds into query tokens through the three-stage
        pipeline: furthest point sampling → Fourier encoding → linear projection.
        
        Args:
            point_cloud: Input point cloud tensor of shape (batch_size, N, 3)
                        where N is the number of input points (variable)
        
        Returns:
            Query tokens tensor of shape (batch_size, num_points, embed_dim)
            where num_points=256 and embed_dim=1536
        
        Raises:
            ValueError: If input tensor has invalid shape or contains invalid values
            RuntimeError: If processing fails
        """
        # Input validation
        if not isinstance(point_cloud, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(point_cloud)}")
        
        if point_cloud.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch_size, N, 3), got {point_cloud.dim()}D tensor")
        
        if point_cloud.size(-1) != 3:
            raise ValueError(f"Expected 3D coordinates (last dim = 3), got {point_cloud.size(-1)}")
        
        batch_size, N, _ = point_cloud.shape
        
        if N == 0:
            raise ValueError("Point cloud cannot be empty")
        
        # Check for invalid values
        if torch.isnan(point_cloud).any():
            raise ValueError("Point cloud contains NaN values")
        
        if torch.isinf(point_cloud).any():
            raise ValueError("Point cloud contains infinite values")
        
        try:
            # Step 1: Furthest Point Sampling
            # Downsample each point cloud in the batch to exactly num_points
            sampled_points = self.furthest_point_sampling(point_cloud, self.num_points)
            
            # Verify sampling output shape
            expected_shape = (batch_size, self.num_points, 3)
            if sampled_points.shape != expected_shape:
                raise RuntimeError(f"FPS output shape mismatch: expected {expected_shape}, got {sampled_points.shape}")
            
            # Step 2: Fourier Positional Encoding
            # Transform 3D coordinates to higher-dimensional representations
            encoded_coords = self.fourier_encoder.encode(sampled_points)
            
            # Verify encoding output shape
            expected_encoded_shape = (batch_size, self.num_points, self.fourier_encoder.get_output_dim())
            if encoded_coords.shape != expected_encoded_shape:
                raise RuntimeError(f"Fourier encoding output shape mismatch: expected {expected_encoded_shape}, got {encoded_coords.shape}")
            
            # Step 3: Linear Projection
            # Map encoded coordinates to LLM embedding space
            query_tokens = self.linear_proj(encoded_coords)
            
            # Verify final output shape
            expected_output_shape = (batch_size, self.num_points, self.embed_dim)
            if query_tokens.shape != expected_output_shape:
                raise RuntimeError(f"Linear projection output shape mismatch: expected {expected_output_shape}, got {query_tokens.shape}")
            
            return query_tokens
            
        except Exception as e:
            logger.error(f"Error in PointCloudProjector forward pass: {e}")
            raise RuntimeError(f"Point cloud projection failed: {e}")
    
    def furthest_point_sampling(
        self, 
        point_cloud: torch.Tensor, 
        num_points: int
    ) -> torch.Tensor:
        """
        Apply furthest point sampling to downsample point clouds.
        
        This method handles batch processing and delegates to the FPS implementation
        in utils/point_cloud_utils.py for consistency across the codebase.
        
        Args:
            point_cloud: Input point cloud tensor of shape (batch_size, N, 3)
            num_points: Target number of points after sampling
        
        Returns:
            Sampled point cloud tensor of shape (batch_size, num_points, 3)
        
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If sampling fails
        """
        if not isinstance(point_cloud, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(point_cloud)}")
        
        if point_cloud.dim() != 3 or point_cloud.size(-1) != 3:
            raise ValueError(f"Expected shape (batch_size, N, 3), got {point_cloud.shape}")
        
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        
        batch_size, N, _ = point_cloud.shape
        
        try:
            # Process each point cloud in the batch
            sampled_batch = []
            
            for i in range(batch_size):
                # Extract single point cloud
                single_pc = point_cloud[i]  # Shape: (N, 3)
                
                # Apply furthest point sampling
                sampled_pc = furthest_point_sampling(single_pc, num_points)
                
                # Verify output shape
                if sampled_pc.shape != (num_points, 3):
                    raise RuntimeError(f"FPS failed for batch item {i}: expected ({num_points}, 3), got {sampled_pc.shape}")
                
                sampled_batch.append(sampled_pc)
            
            # Stack back into batch tensor
            sampled_points = torch.stack(sampled_batch, dim=0)  # Shape: (batch_size, num_points, 3)
            
            # Ensure output is on same device as input
            sampled_points = sampled_points.to(point_cloud.device)
            
            return sampled_points
            
        except Exception as e:
            logger.error(f"Error in furthest point sampling: {e}")
            raise RuntimeError(f"Furthest point sampling failed: {e}")
    
    def get_output_dim(self) -> int:
        """
        Get the output embedding dimension.
        
        Returns:
            Output embedding dimension (embed_dim)
        """
        return self.embed_dim
    
    def get_num_points(self) -> int:
        """
        Get the number of output points after sampling.
        
        Returns:
            Number of sampled points (num_points)
        """
        return self.num_points
    
    def get_fourier_output_dim(self) -> int:
        """
        Get the Fourier encoding output dimension.
        
        Returns:
            Fourier encoding output dimension
        """
        return self.fourier_encoder.get_output_dim()
    
    def validate_input(
        self, 
        point_cloud: torch.Tensor,
        strict: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate input point cloud format and properties.
        
        Args:
            point_cloud: Input point cloud tensor to validate
            strict: If True, apply strict validation criteria
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic type and shape validation
            if not isinstance(point_cloud, torch.Tensor):
                return False, f"Expected torch.Tensor, got {type(point_cloud)}"
            
            if point_cloud.dim() != 3:
                return False, f"Expected 3D tensor, got {point_cloud.dim()}D"
            
            if point_cloud.size(-1) != 3:
                return False, f"Expected 3D coordinates, got {point_cloud.size(-1)}D"
            
            batch_size, N, _ = point_cloud.shape
            
            if N == 0:
                return False, "Point cloud cannot be empty"
            
            # Check for invalid values
            if torch.isnan(point_cloud).any():
                return False, "Point cloud contains NaN values"
            
            if torch.isinf(point_cloud).any():
                return False, "Point cloud contains infinite values"
            
            # Strict validation checks
            if strict:
                # Check if we have enough points for meaningful sampling
                if N < self.num_points:
                    return False, f"Point cloud has {N} points, less than required {self.num_points}"
                
                # Check coordinate ranges (reasonable values)
                pc_min = torch.min(point_cloud).item()
                pc_max = torch.max(point_cloud).item()
                
                if abs(pc_min) > 1000 or abs(pc_max) > 1000:
                    return False, f"Coordinates out of reasonable range: [{pc_min:.2f}, {pc_max:.2f}]"
            
            return True, "Valid input"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def extra_repr(self) -> str:
        """
        Extra representation for model summary and debugging.
        
        Returns:
            String representation of module parameters
        """
        return (f'num_points={self.num_points}, '
                f'embed_dim={self.embed_dim}, '
                f'num_freqs={self.num_freqs}, '
                f'fourier_output_dim={self.get_fourier_output_dim()}')


def create_point_cloud_projector(config: Optional[Dict[str, Any]] = None) -> PointCloudProjector:
    """
    Factory function to create PointCloudProjector with configuration.
    
    Args:
        config: Configuration dictionary from config.yaml
    
    Returns:
        Configured PointCloudProjector instance
    """
    return PointCloudProjector(config=config)


def test_point_cloud_projector():
    """
    Test function to verify PointCloudProjector functionality.
    This function can be used for debugging and validation.
    """
    print("Testing PointCloudProjector...")
    
    # Test configuration
    test_config = {
        'model': {
            'num_points': 256,
            'embedding_dim': 1536,
            'fourier_encoding': {
                'num_freqs': 64
            }
        }
    }
    
    # Create projector
    projector = PointCloudProjector(config=test_config)
    
    # Test with sample input
    batch_size = 2
    num_input_points = 1024
    input_pc = torch.randn(batch_size, num_input_points, 3)
    
    print(f"Input shape: {input_pc.shape}")
    
    # Forward pass
    query_tokens = projector(input_pc)
    print(f"Output shape: {query_tokens.shape}")
    
    # Verify dimensions
    expected_shape = (batch_size, 256, 1536)
    assert query_tokens.shape == expected_shape, f"Expected {expected_shape}, got {query_tokens.shape}"
    
    # Test gradient flow
    input_pc.requires_grad_(True)
    query_tokens = projector(input_pc)
    loss = query_tokens.sum()
    loss.backward()
    
    assert input_pc.grad is