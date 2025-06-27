```python
"""
CAD utilities for CAD-Recode system.
Handles CadQuery operations, code validation, execution, and point cloud conversion.
"""

import ast
import contextlib
import logging
import signal
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cadquery as cq
import numpy as np
import torch
import trimesh
from scipy.spatial.distance import cdist

from config import Config


class TimeoutError(Exception):
    """Custom exception for execution timeout."""
    pass


class CADUtils:
    """
    Utility class handling all CadQuery-related operations.
    Implements CAD code validation, execution, point cloud conversion, and template generation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize CAD utilities with configuration.
        
        Args:
            config: Configuration object with CAD-related parameters
        """
        self.config = config if config is not None else Config()
        self.logger = logging.getLogger(__name__)
        
        # Cache for validated code patterns
        self._code_cache: Dict[str, bool] = {}
        
        # Template cache for performance
        self._template_cache: Dict[str, str] = {}
        
        # Initialize CadQuery environment
        self._setup_cadquery_environment()
    
    def _setup_cadquery_environment(self) -> None:
        """Set up CadQuery execution environment with proper imports."""
        try:
            # Test CadQuery availability
            test_wp = cq.Workplane("XY").box(1, 1, 1)
            self.logger.info("CadQuery environment initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize CadQuery environment: {e}")
            raise RuntimeError(f"CadQuery initialization failed: {e}")
    
    @contextlib.contextmanager
    def _timeout_context(self, timeout_seconds: float = 30.0):
        """
        Context manager for execution timeout.
        
        Args:
            timeout_seconds: Timeout duration in seconds
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout_seconds} seconds")
        
        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            yield
        finally:
            # Restore original handler and cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def validate_cad_code(self, code: str) -> bool:
        """
        Validate CadQuery Python code for syntax and structure.
        
        Args:
            code: Python code string containing CadQuery operations
            
        Returns:
            bool: True if code is valid, False otherwise
        """
        if not isinstance(code, str) or not code.strip():
            return False
        
        # Check cache first
        code_hash = str(hash(code))
        if code_hash in self._code_cache:
            return self._code_cache[code_hash]
        
        try:
            # Parse Python syntax
            parsed_ast = ast.parse(code)
            
            # Check for required imports
            has_cadquery_import = False
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == 'cadquery' or alias.asname == 'cq':
                            has_cadquery_import = True
                elif isinstance(node, ast.ImportFrom):
                    if node.module == 'cadquery':
                        has_cadquery_import = True
            
            # Check for CadQuery operations
            has_workplane = False
            has_operations = False
            
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Attribute):
                    if node.attr in ['Workplane', 'workplane']:
                        has_workplane = True
                    elif node.attr in ['box', 'circle', 'rect', 'extrude', 'cut', 'union']:
                        has_operations = True
                elif isinstance(node, ast.Call):
                    if hasattr(node.func, 'id'):
                        if node.func.id in ['Workplane']:
                            has_workplane = True
            
            # Basic validation: must have CadQuery imports and operations
            is_valid = has_cadquery_import and has_workplane
            
            # Cache result
            self._code_cache[code_hash] = is_valid
            
            return is_valid
            
        except SyntaxError as e:
            self.logger.debug(f"Syntax error in CAD code: {e}")
            self._code_cache[code_hash] = False
            return False
        except Exception as e:
            self.logger.debug(f"Validation error: {e}")
            self._code_cache[code_hash] = False
            return False
    
    def execute_cad_code(self, code: str) -> Optional[cq.Workplane]:
        """
        Execute CadQuery Python code and return resulting workplane.
        
        Args:
            code: Python code string containing CadQuery operations
            
        Returns:
            Optional[cq.Workplane]: Resulting workplane or None if execution fails
        """
        if not self.validate_cad_code(code):
            self.logger.debug("Code validation failed, skipping execution")
            return None
        
        try:
            # Create execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'cadquery': cq,
                'cq': cq,
                'math': __import__('math'),
                'numpy': np,
                'np': np,
            }
            exec_locals = {}
            
            # Execute with timeout
            timeout = getattr(self.config.cadquery, 'execution_timeout', 30.0)
            
            with self._timeout_context(timeout):
                exec(code, exec_globals, exec_locals)
            
            # Find the workplane result
            workplane_result = None
            
            # Look for common variable names
            for var_name in ['result', 'workplane', 'wp', 'model', 'shape']:
                if var_name in exec_locals:
                    candidate = exec_locals[var_name]
                    if isinstance(candidate, cq.Workplane):
                        workplane_result = candidate
                        break
            
            # If no named result found, look for any Workplane object
            if workplane_result is None:
                for value in exec_locals.values():
                    if isinstance(value, cq.Workplane):
                        workplane_result = value
                        break
            
            # Validate result
            if workplane_result is not None:
                # Check if workplane has valid geometry
                try:
                    # Test if we can get the shape
                    shape = workplane_result.val()
                    if shape is not None:
                        return workplane_result
                except Exception as e:
                    self.logger.debug(f"Invalid workplane geometry: {e}")
                    return None
            
            self.logger.debug("No valid workplane found in execution result")
            return None
            
        except TimeoutError:
            self.logger.warning(f"CAD code execution timed out after {timeout} seconds")
            return None
        except Exception as e:
            self.logger.debug(f"CAD code execution failed: {e}")
            return None
    
    def workplane_to_point_cloud(self, workplane: cq.Workplane, num_points: int = 2048) -> torch.Tensor:
        """
        Convert CadQuery workplane to point cloud via surface sampling.
        
        Args:
            workplane: CadQuery workplane object
            num_points: Number of points to sample from surfaces
            
        Returns:
            torch.Tensor: Point cloud with shape [num_points, 3]
            
        Raises:
            ValueError: If workplane is invalid or conversion fails
        """
        if not isinstance(workplane, cq.Workplane):
            raise ValueError("Input must be a CadQuery Workplane object")
        
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        
        try:
            # Export workplane to mesh
            # Get the compound shape from workplane
            shape = workplane.val()
            
            if shape is None:
                raise ValueError("Workplane contains no valid geometry")
            
            # Convert to mesh using CadQuery's mesh export
            # Create a temporary file for mesh export
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Export to STL format
                cq.exporters.export(workplane, tmp_path, cq.exporters.ExportTypes.STL)
                
                # Load mesh using trimesh
                mesh = trimesh.load_mesh(tmp_path)
                
                # Clean up temporary file
                Path(tmp_path).unlink(missing_ok=True)
                
                # Handle different mesh types
                if isinstance(mesh, trimesh.Scene):
                    # If it's a scene, combine all geometries
                    combined_mesh = trimesh.util.concatenate([
                        geom for geom in mesh.geometry.values()
                        if isinstance(geom, trimesh.Trimesh)
                    ])
                    mesh = combined_mesh
                
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError("Failed to create valid mesh from workplane")
                
                # Ensure mesh is watertight for proper sampling
                if not mesh.is_watertight:
                    self.logger.debug("Mesh is not watertight, attempting to fix")
                    mesh.fill_holes()
                
                # Sample points from mesh surface
                if mesh.area > 0:
                    # Use trimesh's surface sampling
                    points, _ = trimesh.sample.sample_surface(mesh, num_points)
                else:
                    # Fallback: sample from vertices if no surface area
                    if len(mesh.vertices) >= num_points:
                        # Subsample vertices
                        indices = np.random.choice(len(mesh.vertices), num_points, replace=False)
                        points = mesh.vertices[indices]
                    else:
                        # Repeat vertices to reach desired count
                        points = mesh.vertices
                        while len(points) < num_points:
                            points = np.vstack([points, mesh.vertices])
                        points = points[:num_points]
                
                # Convert to tensor
                points_tensor = torch.from_numpy(points.astype(np.float32))
                
                # Validate result
                if points_tensor.shape != (num_points, 3):
                    raise ValueError(f"Invalid point cloud shape: {points_tensor.shape}")
                
                # Check for invalid coordinates
                if not torch.isfinite(points_tensor).all():
                    self.logger.warning("Point cloud contains invalid coordinates")
                    # Replace invalid points with valid ones
                    valid_mask = torch.isfinite(points_tensor).all(dim=1)
                    if valid_mask.sum() > 0:
                        valid_points = points_tensor[valid_mask]
                        # Repeat valid points to fill invalid positions
                        invalid_mask = ~valid_mask
                        num_invalid = invalid_mask.sum()
                        if num_invalid > 0:
                            replacement_indices = torch.randint(0, len(valid_points), (num_invalid,))
                            points_tensor[invalid_mask] = valid_points[replacement_indices]
                    else:
                        # All points invalid, return zeros
                        points_tensor = torch.zeros(num_points, 3, dtype=torch.float32)
                
                return points_tensor
                
            except Exception as e:
                # Clean up temporary file in case of error
                Path(tmp_path).unlink(missing_ok=True)
                raise e
                
        except Exception as e:
            self.logger.error(f"Failed to convert workplane to point cloud: {e}")
            raise ValueError(f"Workplane to point cloud conversion failed: {e}")
    
    def normalize_point_cloud(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Normalize point cloud to unit scale centered at origin.
        
        Args:
            pc: Point cloud tensor with shape [N, 3]
            
        Returns:
            torch.Tensor: Normalized point cloud with shape [N, 3]
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
        centroid = torch.mean(normalized_pc, dim=0)
        normalized_pc = normalized_pc - centroid.unsqueeze(0)
        
        # Scale to unit cube
        min_coords = torch.min(normalized_pc, dim=0)[0]
        max_coords = torch.max(normalized_pc, dim=0)[0]
        bbox_size = max_coords - min_coords
        max_dim = torch.max(bbox_size)
        
        # Handle degenerate cases
        if max_dim > 1e-8:
            normalized_pc = normalized_pc / max_dim
        else:
            warnings.warn("Point cloud has very small bounding box, skipping scaling")
        
        return normalized_pc
    
    def generate_rectangle_sketch(self, width: float, height: float, 
                                center: Tuple[float, float] = (0.0, 0.0)) -> str:
        """
        Generate CadQuery code for rectangular sketch.
        
        Args:
            width: Rectangle width
            height: Rectangle height
            center: Center coordinates (x, y)
            
        Returns:
            str: CadQuery code for rectangle sketch
        """
        # Validate parameters against config
        min_size = getattr(self.config.generation, 'min_sketch_size', 0.1)
        max_size = getattr(self.config.generation, 'max_sketch_size', 10.0)
        
        if not (min_size <= width <= max_size):
            raise ValueError(f"Width {width} outside valid range [{min_size}, {max_size}]")
        if not (min_size <= height <= max_size):
            raise ValueError(f"Height {height} outside valid range [{min_size}, {max_size}]")
        
        cx, cy = center
        indentation = getattr(self.config.cadquery, 'indentation', '    ')
        
        code = f"""import cadquery as cq

# Create rectangular sketch
result = (cq.Workplane("XY")
{indentation}.center({cx}, {cy})
{indentation}.rect({width}, {height}))
"""
        return code
    
    def generate_circle_sketch(self, radius: float, 
                             center: Tuple[float, float] = (0.0, 0.0)) -> str:
        """
        Generate CadQuery code for circular sketch.
        
        Args:
            radius: Circle radius
            center: Center coordinates (x, y)
            
        Returns:
            str: CadQuery code for circle sketch
        """
        # Validate parameters
        min_size = getattr(self.config.generation, 'min_sketch_size', 0.1)
        max_size = getattr(self.config.generation, 'max_sketch_size', 10.0)
        
        if not (min_size/2 <= radius <= max_size/2):
            raise ValueError(f"Radius {radius} outside valid range [{min_size/2}, {max_size/2}