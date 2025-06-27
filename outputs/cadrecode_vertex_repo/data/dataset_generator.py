## Code: data/dataset_generator.py

```python
"""
Procedural dataset generation for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the procedural dataset generation pipeline described in 
Section 3.2 and Algorithms 1 & 2 from the paper. It generates 1 million CAD 
sketch-extrude sequences as executable CadQuery Python code with controlled 
features and design patterns.

The generation follows the paper's approach:
- Algorithm 1: Generate2DSketch - Creates valid 2D sketches
- Algorithm 2: GenerateCAD - Produces 3D CAD models from sketches
- Validation framework ensuring φ_syn and φ_cad compliance
"""

import os
import sys
import json
import random
import logging
import hashlib
import tempfile
import subprocess
import time
import traceback
import math
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

import numpy as np

try:
    import cadquery as cq
    from cadquery import exporters
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    warnings.warn("CadQuery not available. Dataset generation will be limited.")

try:
    from OCC.Core import BRepCheck_Analyzer
    from OCC.Core import TopoDS_Shape
    PYTHONIC_OCC_AVAILABLE = True
except ImportError:
    PYTHONIC_OCC_AVAILABLE = False
    warnings.warn("PythonOCC not available. Advanced geometric validation disabled.")

# Import validation utilities
from utils.cad_validation import CADValidator, ValidationError, ValidationTimeout


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics tracking for dataset generation process."""
    total_attempts: int = 0
    successful_generations: int = 0
    syntax_failures: int = 0
    semantic_failures: int = 0
    geometric_failures: int = 0
    duplicate_rejections: int = 0
    timeout_failures: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class SketchMetadata:
    """Metadata for generated sketch components."""
    primitives_count: int
    primitive_types: List[str]
    operations_used: List[str]
    boundary_loops: int
    complexity_score: float


@dataclass
class CADModelMetadata:
    """Metadata for generated CAD models."""
    sketch_count: int
    total_primitives: int
    operations_count: int
    planes_used: List[str]
    extrusion_heights: List[float]
    complexity_score: float
    generation_time: float
    code_length: int


class DatasetGenerator:
    """
    Procedural CAD dataset generator implementing Algorithms 1 & 2 from the paper.
    
    Generates 1 million CAD sketch-extrude sequences as CadQuery Python code with
    controlled features, design patterns, and validation to ensure code quality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset generator with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        # Store configuration
        self.config = config
        
        # Extract procedural dataset configuration
        proc_config = config.get('data', {}).get('procedural_dataset', {})
        self.dataset_size = proc_config.get('size', 1000000)
        self.num_primitives_range = proc_config.get('num_primitives_range', [3, 8])
        self.coordinate_range = proc_config.get('coordinate_range', [-100, 100])
        self.coordinate_resolution = proc_config.get('coordinate_resolution', 1)
        
        # CAD configuration
        cad_config = config.get('cadquery', {})
        self.library_import = cad_config.get('library_import', 'import cadquery as cq')
        self.validation_timeout = cad_config.get('validation_timeout', 30)
        
        # System configuration
        system_config = config.get('system', {})
        self.random_seed = system_config.get('random_seed', 42)
        
        # Paths configuration
        paths_config = config.get('paths', {})
        self.data_dir = Path(paths_config.get('data_dir', './data'))
        self.output_dir = self.data_dir / 'procedural_dataset'
        
        # Initialize random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Initialize validation
        self.validator = CADValidator(config.get('cadquery', {}))
        
        # Initialize statistics
        self.stats = GenerationStats()
        
        # Duplicate detection
        self._generated_hashes: Set[str] = set()
        self._code_cache: Dict[str, str] = {}
        
        # Geometric primitives and operations
        self._initialize_generation_parameters()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetGenerator initialized:")
        logger.info(f"  Target size: {self.dataset_size:,}")
        logger.info(f"  Primitives range: {self.num_primitives_range}")
        logger.info(f"  Coordinate range: {self.coordinate_range}")
        logger.info(f"  Resolution: {self.coordinate_resolution}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def _initialize_generation_parameters(self) -> None:
        """Initialize parameters for geometric generation."""
        # Primitive types available for sketch generation
        self.primitive_types = ['Circle', 'RotatedRectangle']
        
        # Boolean operations for combining primitives
        self.boolean_operations = ['Union', 'Cut']
        
        # Canonical planes for workplane generation
        self.canonical_planes = [
            ('XY', (0, 0, 1)),  # (plane_name, normal_vector)
            ('XZ', (0, 1, 0)),
            ('YZ', (1, 0, 0))
        ]
        
        # Extrusion height ranges (quantized)
        self.extrusion_height_range = [5, 50]  # Reasonable range for extrusions
        
        # Size ranges for primitives
        self.circle_radius_range = [3, 25]
        self.rectangle_size_range = [5, 30]
        
        # Plane translation ranges
        self.plane_translation_range = [-20, 20]
        
        # Complexity scoring weights
        self.complexity_weights = {
            'primitives': 1.0,
            'operations': 0.5,
            'sketches': 2.0,
            'planes': 1.5
        }
    
    def generate_sketch(self) -> Dict[str, Any]:
        """
        Generate a 2D sketch following Algorithm 1 from the paper.
        
        Implements Generate2DSketch algorithm:
        1. Choose random number of primitives (3-8)
        2. Build composite shape by combining primitives
        3. Extract boundary loops and analyze topology
        4. Validate shape topology
        
        Returns:
            Dictionary containing sketch data and metadata
            
        Raises:
            ValueError: If sketch generation fails
        """
        try:
            # Step 1: Choose random number of shape primitives (3-8)
            num_primitives = random.randint(*self.num_primitives_range)
            
            # Initialize sketch data
            sketch_data = {
                'primitives': [],
                'operations': [],
                'boundary_components': [],
                'metadata': None
            }
            
            # Track primitives and operations for modularity
            primitive_types_used = []
            operations_used = []
            
            # Step 2: Build shape by combining primitives
            composite_shape = None
            
            for i in range(num_primitives):
                # Choose random primitive type
                primitive_type = random.choice(self.primitive_types)
                primitive_types_used.append(primitive_type)
                
                # Generate primitive parameters
                if primitive_type == 'Circle':
                    primitive_data = self._generate_circle_primitive()
                elif primitive_type == 'RotatedRectangle':
                    primitive_data = self._generate_rectangle_primitive()
                else:
                    raise ValueError(f"Unknown primitive type: {primitive_type}")
                
                # Choose boolean operation (Union adds, Cut subtracts)
                if i == 0:
                    # First primitive is always added
                    boolean_operation = 'Union'
                else:
                    boolean_operation = random.choice(self.boolean_operations)
                
                operations_used.append(boolean_operation)
                
                # Store primitive and operation data
                sketch_data['primitives'].append({
                    'type': primitive_type,
                    'parameters': primitive_data,
                    'operation': boolean_operation
                })
                sketch_data['operations'].append(boolean_operation)
                
                # Apply operation to composite shape (conceptual - actual geometry later)
                composite_shape = self._apply_primitive_operation(
                    composite_shape, primitive_data, boolean_operation
                )
            
            # Step 3: Extract boundary loops (simplified for code generation)
            boundary_components = self._extract_boundary_components(sketch_data['primitives'])
            sketch_data['boundary_components'] = boundary_components
            
            # Step 4: Validate shape topology
            if not self._validate_shape_topology(boundary_components):
                raise ValueError("Invalid shape topology generated")
            
            # Calculate complexity score
            complexity_score = self._calculate_sketch_complexity(
                num_primitives, len(set(primitive_types_used)), len(boundary_components)
            )
            
            # Create metadata
            sketch_data['metadata'] = SketchMetadata(
                primitives_count=num_primitives,
                primitive_types=primitive_types_used,
                operations_used=operations_used,
                boundary_loops=len(boundary_components),
                complexity_score=complexity_score
            )
            
            return sketch_data
            
        except Exception as e:
            logger.error(f"Error generating sketch: {e}")
            raise ValueError(f"Sketch generation failed: {e}")
    
    def _generate_circle_primitive(self) -> Dict[str, Any]:
        """Generate parameters for a circle primitive."""
        # Quantized center coordinates
        center_x = self._quantize_coordinate(
            random.uniform(*self.coordinate_range)
        )
        center_y = self._quantize_coordinate(
            random.uniform(*self.coordinate_range)
        )
        
        # Quantized radius
        radius = self._quantize_coordinate(
            random.uniform(*self.circle_radius_range)
        )
        
        # Ensure positive radius
        radius = max(radius, self.coordinate_resolution)
        
        return {
            'center': (center_x, center_y),
            'radius': radius
        }
    
    def _generate_rectangle_primitive(self) -> Dict[str, Any]:
        """Generate parameters for a rotated rectangle primitive."""
        # Quantized center coordinates
        center_x = self._quantize_coordinate(
            random.uniform(*self.coordinate_range)
        )
        center_y = self._quantize_coordinate(
            random.uniform(*self.coordinate_range)
        )
        
        # Quantized dimensions
        width = self._quantize_coordinate(
            random.uniform(*self.rectangle_size_range)
        )
        height = self._quantize_coordinate(
            random.uniform(*self.rectangle_size_range)
        )
        
        # Ensure positive dimensions
        width = max(width, self.coordinate_resolution)
        height = max(height, self.coordinate_resolution)
        
        # Random rotation angle (quantized to degrees)
        rotation = random.randint(0, 359)
        
        return {
            'center': (center_x, center_y),
            'width': width,
            'height': height,
            'rotation': rotation
        }
    
    def _quantize_coordinate(self, value: float) -> int:
        """
        Quantize coordinate to specified resolution.
        
        Args:
            value: Floating point coordinate value
            
        Returns:
            Quantized integer coordinate
        """
        # Round to nearest resolution unit
        quantized = round(value / self.coordinate_resolution) * self.coordinate_resolution
        
        # Clamp to coordinate range
        min_coord, max_coord = self.coordinate_range
        quantized = max(min_coord, min(max_coord, quantized))
        
        return int(quantized)
    
    def _apply_primitive_operation(
        self, 
        composite_shape: Optional[Any], 
        primitive_data: Dict[str, Any], 
        operation: str
    ) -> Any:
        """
        Apply primitive operation to composite shape (conceptual).
        
        This is a simplified version for tracking - actual geometry
        is generated during code creation.
        """
        # For now, just track the operation conceptually
        # Real geometric operations happen during code generation
        if composite_shape is None:
            return {'primitives': [primitive_data], 'operations': [operation]}
        else:
            composite_shape['primitives'].append(primitive_data)
            composite_shape['operations'].append(operation)
            return composite_shape
    
    def _extract_boundary_components(
        self, 
        primitives: List[Dict[str, Any]]
    ) -> List[Tuple[List[Dict[str, Any]], bool]]:
        """
        Extract boundary components from primitive list.
        
        Returns list of (edge_sequence, is_outer) tuples.
        """
        boundary_components = []
        
        # For each primitive, create boundary representation
        for i, primitive in enumerate(primitives):
            if primitive['type'] == 'Circle':
                # Circle creates single closed loop
                edges = [{'type': 'circle', 'parameters': primitive['parameters']}]
                is_outer = primitive['operation'] == 'Union'
                boundary_components.append((edges, is_outer))
                
            elif primitive['type'] == 'RotatedRectangle':
                # Rectangle creates four line segments
                edges = self._rectangle_to_edges(primitive['parameters'])
                is_outer = primitive['operation'] == 'Union'
                boundary_components.append((edges, is_outer))
        
        return boundary_components
    
    def _rectangle_to_edges(self, rect_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert rectangle parameters to edge list."""
        center_x, center_y = rect_params['center']
        width = rect_params['width']
        height = rect_params['height']
        rotation = rect_params.get('rotation', 0)
        
        # Calculate corner points (before rotation)
        half_w = width / 2
        half_h = height / 2
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        # Apply rotation and translation
        rot_rad = math.radians(rotation)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        
        transformed_corners = []
        for x, y in corners:
            # Rotate
            new_x = x * cos_r - y * sin_r
            new_y = x * sin_r + y * cos_r
            # Translate
            new_x += center_x
            new_y += center_y
            # Quantize
            new_x = self._quantize_coordinate(new_x)
            new_y = self._quantize_coordinate(new_y)
            transformed_corners.append((new_x, new_y))
        
        # Create line segments
        edges = []
        for i in range(4):
            start