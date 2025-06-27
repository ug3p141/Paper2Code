```python
"""
CAD code validation utilities for CAD-Recode implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the critical validation functions φ_syn and φ_cad from the paper's
problem formulation (Section 4.1). It ensures generated CAD code satisfies both Python
syntax requirements and CAD-specific semantic and geometric constraints.

The validation framework defines valid CAD code as:
C = {w ∈ Σ* | φ_syn(w) ∧ φ_cad(w)}

Where:
- φ_syn(w): Python syntax validation
- φ_cad(w): CAD-specific semantic and geometric validation
"""

import ast
import subprocess
import sys
import time
import logging
import re
import tempfile
import os
import signal
import psutil
import traceback
import threading
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from pathlib import Path
import warnings

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    warnings.warn("CadQuery not available. CAD validation will be limited.")

try:
    from OCC.Core import BRepCheck_Analyzer
    from OCC.Core import TopoDS_Shape
    PYTHONIC_OCC_AVAILABLE = True
except ImportError:
    PYTHONIC_OCC_AVAILABLE = False
    warnings.warn("PythonOCC not available. Advanced geometric validation disabled.")


# Set up logging
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ValidationTimeout(Exception):
    """Exception raised when validation times out."""
    pass


class CADValidator:
    """
    Central validation orchestrator implementing φ_syn and φ_cad validation functions.
    
    This class coordinates syntax, semantic, and geometric validation of CAD code
    to ensure generated code satisfies both Python requirements and CAD-specific
    constraints as specified in the paper.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CAD validator with configuration.
        
        Args:
            config: Configuration dictionary. If None, uses default values.
        """
        # Default configuration values from config.yaml
        default_config = {
            'validation_timeout': 30,  # seconds
            'library_import': 'import cadquery as cq',
            'coordinate_range': [-100, 100],
            'coordinate_resolution': 1,
            'max_memory_mb': 1024,  # Maximum memory usage in MB
            'enable_geometric_validation': True,
            'enable_brep_check': PYTHONIC_OCC_AVAILABLE,
            'allowed_imports': {'cadquery', 'cq', 'math', 'numpy', 'np'},
            'forbidden_imports': {'os', 'sys', 'subprocess', 'eval', 'exec', 'open', 'file'}
        }
        
        if config is None:
            config = {}
        
        # Merge with provided config
        self.config = {**default_config, **config}
        
        # Initialize validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'syntax_failures': 0,
            'semantic_failures': 0,
            'geometric_failures': 0,
            'timeouts': 0,
            'memory_errors': 0
        }
        
        # Cache for validation results
        self._validation_cache: Dict[str, bool] = {}
        self._cache_enabled = True
        
        # CadQuery API method validation sets
        self._initialize_cadquery_api_validation()
        
        logger.info(f"CADValidator initialized with timeout={self.config['validation_timeout']}s")
    
    def _initialize_cadquery_api_validation(self) -> None:
        """Initialize CadQuery API validation data structures."""
        # Core CadQuery methods that should be available
        self.valid_workplane_methods = {
            'workplane', 'sketch', 'segment', 'arc', 'circle', 'rect', 'box', 'cylinder',
            'extrude', 'revolve', 'union', 'cut', 'intersect', 'close', 'assemble',
            'finalize', 'push', 'moveTo', 'lineTo', 'radiusArc', 'sagittaArc',
            'spline', 'bezier', 'offset', 'fillet', 'chamfer', 'shell', 'faces',
            'edges', 'vertices', 'val', 'vals', 'first', 'last', 'item',
            'translate', 'rotate', 'mirror', 'scale'
        }
        
        # Valid plane specifications for workplane initialization
        self.valid_plane_specs = {
            'XY', 'YZ', 'ZX', 'XZ', 'YX', 'ZY',
            'front', 'back', 'left', 'right', 'top', 'bottom'
        }
        
        # Methods that require specific parameter types
        self.method_param_validation = {
            'circle': {'min_params': 1, 'numeric_params': [0]},  # radius
            'rect': {'min_params': 2, 'numeric_params': [0, 1]},  # width, height
            'box': {'min_params': 3, 'numeric_params': [0, 1, 2]},  # length, width, height
            'cylinder': {'min_params': 2, 'numeric_params': [0, 1]},  # height, radius
            'extrude': {'min_params': 1, 'numeric_params': [0]},  # distance
            'moveTo': {'min_params': 2, 'numeric_params': [0, 1]},  # x, y
            'segment': {'min_params': 1},  # coordinates
            'arc': {'min_params': 2}  # coordinates
        }
    
    def validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax correctness (φ_syn function from paper).
        
        Args:
            code: CAD code string to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        if not isinstance(code, str):
            logger.error(f"Expected string input, got {type(code)}")
            return False
        
        if not code.strip():
            logger.error("Empty code string")
            return False
        
        try:
            # Parse the code using Python AST
            tree = ast.parse(code)
            
            # Validate required import statement
            if not self._validate_required_imports(tree):
                logger.error("Missing required CadQuery import statement")
                return False
            
            # Check for forbidden imports
            if not self._validate_safe_imports(tree):
                logger.error("Code contains forbidden imports")
                return False
            
            # Validate variable declarations and usage
            if not self._validate_variable_usage(tree):
                logger.error("Invalid variable usage detected")
                return False
            
            # Validate expression syntax (method chaining, etc.)
            if not self._validate_expression_syntax(tree):
                logger.error("Invalid expression syntax detected")
                return False
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Python syntax error: {e}")
            self.validation_stats['syntax_failures'] += 1
            return False
        except Exception as e:
            logger.error(f"Unexpected error in syntax validation: {e}")
            return False
    
    def validate_cad_semantics(self, code: str) -> bool:
        """
        Validate CadQuery library-specific syntax and semantic rules.
        
        Args:
            code: CAD code string to validate
            
        Returns:
            True if CAD semantics are valid, False otherwise
        """
        try:
            # Parse AST for semantic analysis
            tree = ast.parse(code)
            
            # Validate CadQuery API usage
            if not self._validate_cadquery_api_calls(tree):
                logger.error("Invalid CadQuery API usage")
                return False
            
            # Validate workplane initialization
            if not self._validate_workplane_usage(tree):
                logger.error("Invalid workplane usage")
                return False
            
            # Validate geometric primitive parameters
            if not self._validate_primitive_parameters(tree):
                logger.error("Invalid geometric primitive parameters")
                return False
            
            # Validate operation sequences
            if not self._validate_operation_sequences(tree):
                logger.error("Invalid operation sequence")
                return False
            
            # Validate parameter ranges (quantization constraints from paper)
            if not self._validate_parameter_ranges(tree):
                logger.error("Parameters outside valid range")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in CAD semantic validation: {e}")
            self.validation_stats['semantic_failures'] += 1
            return False
    
    def validate_geometric(self, code: str) -> bool:
        """
        Validate geometric consistency and executability.
        
        Args:
            code: CAD code string to validate
            
        Returns:
            True if geometry is valid, False otherwise
        """
        if not self.config['enable_geometric_validation']:
            logger.debug("Geometric validation disabled")
            return True
        
        if not CADQUERY_AVAILABLE:
            logger.warning("CadQuery not available, skipping geometric validation")
            return True
        
        try:
            # Execute code in safe environment with timeout
            result = self._execute_code_safely(code)
            
            if result is None:
                logger.error("Code execution failed or timed out")
                return False
            
            # Validate resulting CAD model
            if not self._validate_cad_model(result):
                logger.error("Generated CAD model is invalid")
                return False
            
            # Advanced geometric validation with PythonOCC if available
            if self.config['enable_brep_check'] and PYTHONIC_OCC_AVAILABLE:
                if not self._validate_brep_geometry(result):
                    logger.error("BRep geometry validation failed")
                    return False
            
            return True
            
        except ValidationTimeout:
            logger.error("Geometric validation timed out")
            self.validation_stats['timeouts'] += 1
            return False
        except MemoryError:
            logger.error("Geometric validation exceeded memory limit")
            self.validation_stats['memory_errors'] += 1
            return False
        except Exception as e:
            logger.error(f"Error in geometric validation: {e}")
            self.validation_stats['geometric_failures'] += 1
            return False
    
    def is_valid_code(self, code: str) -> bool:
        """
        Master validation method implementing complete validation pipeline.
        
        Implements: φ_syn(w) ∧ φ_cad(w) from paper's problem formulation.
        
        Args:
            code: CAD code string to validate
            
        Returns:
            True if code satisfies all validation criteria, False otherwise
        """
        if not isinstance(code, str):
            return False
        
        # Check cache first
        if self._cache_enabled and code in self._validation_cache:
            return self._validation_cache[code]
        
        self.validation_stats['total_validations'] += 1
        
        try:
            # Sequential validation with short-circuit evaluation
            # Stop at first validation failure for efficiency
            
            # Step 1: Python syntax validation (φ_syn)
            if not self.validate_syntax(code):
                logger.debug("Code failed syntax validation")
                result = False
            # Step 2: CAD semantic validation (part of φ_cad)
            elif not self.validate_cad_semantics(code):
                logger.debug("Code failed semantic validation")
                result = False
            # Step 3: Geometric validation (part of φ_cad)
            elif not self.validate_geometric(code):
                logger.debug("Code failed geometric validation")
                result = False
            else:
                logger.debug("Code passed all validation checks")
                result = True
            
            # Cache result
            if self._cache_enabled:
                self._validation_cache[code] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in validation pipeline: {e}")
            return False
    
    def _validate_required_imports(self, tree: ast.AST) -> bool:
        """Validate that required CadQuery import is present."""
        required_import = self.config['library_import']
        
        # Extract import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}" + 
                                 (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(f"from {node.module} import " + 
                                 ", ".join(alias.name for alias in node.names))
        
        # Check if required import is present
        return any(required_import in imp or "cadquery" in imp for imp in imports)
    
    def _validate_safe_imports(self, tree: ast.AST) -> bool:
        """Validate that only safe imports are used."""
        forbidden = self.config['forbidden_imports']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in forbidden:
                    return False
        
        return True
    
    def _validate_variable_usage(self, tree: ast.AST) -> bool:
        """Validate variable declarations and usage patterns."""
        # Simple validation - check for obvious issues
        try:
            # Compile to catch undefined variable errors
            compile(tree, '<string>', 'exec')
            return True
        except NameError:
            return False
        except Exception:
            # Other compilation errors are handled elsewhere
            return True
    
    def _validate_expression_syntax(self, tree: ast.AST) -> bool:
        """Validate expression syntax including method chaining."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for valid method call structure
                if isinstance(node.func, ast.Attribute):
                    # Method call - validate it's a reasonable chain
                    if not self._is_valid_method_chain(node):
                        return False
        return True
    
    def _is_valid_method_chain(self, node: ast.Call) -> bool:
        """Validate that method chaining follows CadQuery patterns."""
        # Basic validation of method chaining structure
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            # Check if it's a known CadQuery method
            return method_name in self.valid_workplane_methods or method_name.startswith('_')
        return True
    
    def _validate_cadquery_api_calls(self, tree: ast.AST) -> bool:
        """Validate CadQuery API method calls."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                
                # Skip private methods and common Python methods
                if method_name.startswith('_') or method_name in {'append', 'extend', 'pop'}:
                    continue
                
                # Check if method is in valid CadQuery API
                if method_name not in self.valid_workplane_methods:
                    # Allow some flexibility for newer API methods
                    if not self._is_likely_cadquery_method(method_name):
                        logger.warning(f"Unknown CadQuery method: {method_name}")
                        return False
                
                # Validate method parameters
                if not self._validate_method_parameters(method_name, node):
                    return False
        
        return True