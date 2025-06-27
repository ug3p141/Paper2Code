```python
"""
Dataset generator for CAD-Recode system.
Procedurally generates 1 million CAD sketch-extrude sequences as Python CadQuery code 
with corresponding point clouds for training.
"""

import json
import logging
import os
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cadquery as cq
import numpy as np
import torch

from config import Config
from utils.cad_utils import CADUtils
from utils.point_cloud_utils import PointCloudUtils


class DatasetGenerator:
    """
    Procedural generator for CAD sketch-extrude sequences and corresponding point clouds.
    Creates training dataset of 1 million samples with geometric and topological validation.
    """
    
    def __init__(self, output_dir: str, num_samples: int, config: Optional[Config] = None):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Directory to save generated dataset
            num_samples: Number of CAD sequences to generate
            config: Configuration object with generation parameters
        """
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.config = config if config is not None else Config()
        
        # Initialize utilities
        self.cad_utils = CADUtils(self.config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Generation statistics
        self.stats = {
            'total_attempts': 0,
            'successful_samples': 0,
            'validation_failures': 0,
            'execution_failures': 0,
            'point_cloud_failures': 0,
            'sketch_type_counts': {
                'rectangle': 0,
                'circle': 0,
                'polygon': 0,
                'complex': 0
            },
            'complexity_distribution': {
                'simple': 0,      # 1 sketch, 1 extrude
                'medium': 0,      # 2-3 sketches
                'complex': 0      # 4-5 sketches
            }
        }
        
        # Create output directories
        self._setup_directories()
        
        # Validate configuration
        self._validate_config()
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        self.logger.info(f"Initialized DatasetGenerator: output_dir={output_dir}, "
                        f"num_samples={num_samples}")
    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.point_clouds_dir = self.output_dir / "point_clouds"
        self.cad_codes_dir = self.output_dir / "cad_codes"
        self.metadata_dir = self.output_dir / "metadata"
        
        for directory in [self.point_clouds_dir, self.cad_codes_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output directories in {self.output_dir}")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters for dataset generation."""
        gen_config = self.config.generation
        
        # Validate sketch size constraints
        if gen_config.min_sketch_size >= gen_config.max_sketch_size:
            raise ValueError("min_sketch_size must be less than max_sketch_size")
        
        # Validate extrude distance constraints  
        if gen_config.min_extrude_distance >= gen_config.max_extrude_distance:
            raise ValueError("min_extrude_distance must be less than max_extrude_distance")
        
        # Validate positive parameters
        for param_name in ['max_sketches_per_model', 'max_extrudes_per_sketch', 'points_per_surface']:
            param_value = getattr(gen_config, param_name)
            if param_value <= 0:
                raise ValueError(f"{param_name} must be positive")
        
        self.logger.info("Configuration validation passed")
    
    def generate_cad_sequence(self) -> str:
        """
        Generate a random CAD sketch-extrude sequence as CadQuery Python code.
        
        Returns:
            str: Valid CadQuery Python code for sketch-extrude sequence
            
        Raises:
            ValueError: If generation fails after multiple attempts
        """
        gen_config = self.config.generation
        
        # Determine model complexity
        num_sketches = random.randint(1, gen_config.max_sketches_per_model)
        
        # Update complexity statistics
        if num_sketches == 1:
            self.stats['complexity_distribution']['simple'] += 1
        elif num_sketches <= 3:
            self.stats['complexity_distribution']['medium'] += 1
        else:
            self.stats['complexity_distribution']['complex'] += 1
        
        # Generate code components
        code_lines = []
        
        # Standard imports
        code_lines.append("import cadquery as cq")
        code_lines.append("")
        
        # Initialize workplane
        workplane_var = "result"
        code_lines.append(f"# Create CAD model with {num_sketches} sketch(es)")
        code_lines.append(f'{workplane_var} = cq.Workplane("XY")')
        
        # Generate sketches and extrudes
        for sketch_idx in range(num_sketches):
            sketch_code, sketch_type = self._generate_sketch_code(sketch_idx)
            code_lines.extend(sketch_code)
            
            # Update sketch type statistics
            self.stats['sketch_type_counts'][sketch_type] += 1
            
            # Generate extrude operations for this sketch
            num_extrudes = random.randint(1, gen_config.max_extrudes_per_sketch)
            
            for extrude_idx in range(num_extrudes):
                extrude_code = self._generate_extrude_code(sketch_idx, extrude_idx)
                code_lines.extend(extrude_code)
        
        # Join code lines
        indentation = self.config.cadquery.indentation
        full_code = "\n".join(code_lines)
        
        return full_code
    
    def _generate_sketch_code(self, sketch_idx: int) -> Tuple[List[str], str]:
        """
        Generate code for a single sketch.
        
        Args:
            sketch_idx: Index of the sketch in the sequence
            
        Returns:
            Tuple[List[str], str]: (code_lines, sketch_type)
        """
        gen_config = self.config.generation
        indentation = self.config.cadquery.indentation
        
        # Choose sketch type randomly
        sketch_types = ['rectangle', 'circle', 'polygon']
        sketch_type = random.choice(sketch_types)
        
        code_lines = []
        code_lines.append(f"")
        code_lines.append(f"# Sketch {sketch_idx + 1}: {sketch_type}")
        
        if sketch_idx == 0:
            # First sketch starts from the base workplane
            base_var = "result"
        else:
            # Subsequent sketches start from faces of previous geometry
            face_selector = random.choice([">Z", "<Z", ">X", "<X", ">Y", "<Y"])
            code_lines.append(f"result = result.faces(\"{face_selector}\").workplane()")
        
        if sketch_type == 'rectangle':
            width = random.uniform(gen_config.min_sketch_size, gen_config.max_sketch_size)
            height = random.uniform(gen_config.min_sketch_size, gen_config.max_sketch_size)
            
            # Ensure reasonable aspect ratio (max 1:5)
            if width / height > 5.0:
                height = width / 5.0
            elif height / width > 5.0:
                width = height / 5.0
            
            # Random center position
            center_x = random.uniform(-1.0, 1.0)
            center_y = random.uniform(-1.0, 1.0)
            
            code_lines.append(f"result = result.center({center_x:.3f}, {center_y:.3f})")
            code_lines.append(f"result = result.rect({width:.3f}, {height:.3f})")
            
        elif sketch_type == 'circle':
            radius = random.uniform(gen_config.min_sketch_size / 2, gen_config.max_sketch_size / 2)
            
            # Random center position
            center_x = random.uniform(-1.0, 1.0)
            center_y = random.uniform(-1.0, 1.0)
            
            code_lines.append(f"result = result.center({center_x:.3f}, {center_y:.3f})")
            code_lines.append(f"result = result.circle({radius:.3f})")
            
        elif sketch_type == 'polygon':
            num_sides = random.randint(3, 8)
            diameter = random.uniform(gen_config.min_sketch_size, gen_config.max_sketch_size)
            
            # Random center position
            center_x = random.uniform(-1.0, 1.0)
            center_y = random.uniform(-1.0, 1.0)
            
            code_lines.append(f"result = result.center({center_x:.3f}, {center_y:.3f})")
            code_lines.append(f"result = result.polygon({num_sides}, {diameter:.3f})")
        
        return code_lines, sketch_type
    
    def _generate_extrude_code(self, sketch_idx: int, extrude_idx: int) -> List[str]:
        """
        Generate code for extrude operation.
        
        Args:
            sketch_idx: Index of the sketch
            extrude_idx: Index of the extrude operation
            
        Returns:
            List[str]: Code lines for extrude operation
        """
        gen_config = self.config.generation
        
        # Generate extrude distance
        extrude_distance = random.uniform(
            gen_config.min_extrude_distance, 
            gen_config.max_extrude_distance
        )
        
        # Choose extrude type
        extrude_types = ['extrude', 'cut', 'union']
        
        # First extrude is always normal extrude
        if sketch_idx == 0 and extrude_idx == 0:
            extrude_type = 'extrude'
        else:
            # Later operations can be boolean operations
            extrude_type = random.choice(extrude_types)
        
        code_lines = []
        code_lines.append(f"")
        code_lines.append(f"# Extrude operation {extrude_idx + 1}")
        
        if extrude_type == 'extrude':
            code_lines.append(f"result = result.extrude({extrude_distance:.3f})")
        elif extrude_type == 'cut':
            # Create a cutting geometry
            code_lines.append(f"cut_shape = result.extrude({extrude_distance:.3f})")
            code_lines.append(f"result = result.cut(cut_shape)")
        elif extrude_type == 'union':
            # Create a union geometry
            code_lines.append(f"union_shape = result.extrude({extrude_distance:.3f})")
            code_lines.append(f"result = result.union(union_shape)")
        
        return code_lines
    
    def execute_cad_code(self, code: str) -> Optional[cq.Workplane]:
        """
        Execute CadQuery Python code and return resulting workplane.
        
        Args:
            code: Python code string containing CadQuery operations
            
        Returns:
            Optional[cq.Workplane]: Resulting workplane or None if execution fails
        """
        try:
            return self.cad_utils.execute_cad_code(code)
        except Exception as e:
            self.logger.debug(f"CAD code execution failed: {e}")
            self.stats['execution_failures'] += 1
            return None
    
    def sample_point_cloud(self, workplane: cq.Workplane) -> torch.Tensor:
        """
        Sample point cloud from CadQuery workplane.
        
        Args:
            workplane: CadQuery workplane object
            
        Returns:
            torch.Tensor: Point cloud with shape [num_points, 3]
            
        Raises:
            ValueError: If point cloud generation fails
        """
        try:
            # Use CADUtils to convert workplane to point cloud
            num_points = self.config.generation.points_per_surface
            point_cloud = self.cad_utils.workplane_to_point_cloud(workplane, num_points)
            
            # Normalize point cloud
            normalized_pc = PointCloudUtils.normalize_point_cloud(point_cloud)
            
            return normalized_pc
            
        except Exception as e:
            self.logger.debug(f"Point cloud sampling failed: {e}")
            self.stats['point_cloud_failures'] += 1
            raise ValueError(f"Point cloud generation failed: {e}")
    
    def _validate_geometry(self, workplane: cq.Workplane) -> bool:
        """
        Validate geometric properties of generated CAD model.
        
        Args:
            workplane: CadQuery workplane to validate
            
        Returns:
            bool: True if geometry is valid, False otherwise
        """
        try:
            # Check if workplane contains valid geometry
            shape = workplane.val()
            if shape is None:
                return False
            
            # Check bounding box
            bbox = workplane.val().BoundingBox()
            
            # Ensure reasonable size
            max_dim = max(bbox.xlen, bbox.ylen, bbox.zlen)
            min_dim = min(bbox.xlen, bbox.ylen, bbox.zlen)
            
            gen_config = self.config.generation
            if max_dim > gen_config.max_sketch_size * 10:  # Too large
                return False
            if min_dim < gen_config.min_sketch_size / 10:  # Too small
                return False
            
            # Check volume (should be positive)
            try:
                volume = workplane.val().Volume()
                if volume <= 0:
                    return False
            except:
                # Some operations might not have volume (e.g., surfaces)
                pass
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Geometry validation failed: {e}")
            return False
    
    def _save_data_pair(self, cad_code: str, point_cloud: torch.Tensor, sample_idx: int) -> None:
        """
        Save CAD code and point cloud pair to files.
        
        Args:
            cad_code: Generated CadQuery Python code
            point_cloud: Corresponding point cloud tensor
            sample_idx: Sample index for file naming
        """
        # Generate file names with zero-padded indices
        file_prefix = f"sample_{sample_idx:06d}"
        
        # Save CAD code
        code_file = self.cad_codes_dir / f"{file_prefix}.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(cad_code)
        
        # Save point cloud
        pc_file = self.point_clouds_dir / f"{file_prefix}.pt"
        torch.save(point_cloud, pc_file)
        
        self.logger.debug(f"Saved sample {sample_idx}: {code_file.name}, {pc_file.name}")
    
    def _save_metadata(self) -> None:
        """Save generation metadata an