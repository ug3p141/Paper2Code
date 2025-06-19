"""
utils.py

This module provides utility functions and configuration parsing for the CAD-Recode project.
It includes functions for:
  - Loading a YAML configuration file.
  - Performing furthest point sampling on point clouds.
  - Computing Fourier positional encoding for 3D coordinates.
  - Validating CAD code through Python syntax checking and CAD geometric validation.
  - Logging messages in a standardized format.
  - Duplicate detection via MD5 hashing.
  - Sampling a point cloud from a generated CAD model.
  - Loading an existing dataset from a file.

Dependencies:
  - numpy==1.21.0
  - torch==1.9.0
  - PyYAML (for configuration parsing)
  - cadquery
  - pythonocc-core
  - tqdm
  - logging
  - hashlib
  - json

Author: Your Name
Date: Today's Date
"""

import os
import sys
import yaml
import json
import logging
import hashlib
import random
import numpy as np
from math import pi
from typing import Any, Dict, List, Set

import cadquery as cq  # for CAD code execution and validations

# -----------------------------------------------------------------------------
# Configuration Parsing Utility
# -----------------------------------------------------------------------------
def load_config(filepath: str = "config.yaml") -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        filepath (str): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.
    """
    if not os.path.exists(filepath):
        logging.error(f"Configuration file {filepath} not found.")
        sys.exit(1)
    try:
        with open(filepath, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error parsing configuration file {filepath}: {e}")
        sys.exit(1)

    # Set default values for missing keys
    config.setdefault("training", {})
    config["training"].setdefault("learning_rate", 0.0002)
    config["training"].setdefault("weight_decay", 0.01)
    config["training"].setdefault("total_iterations", 100000)
    config["training"].setdefault("warmup_iterations", 1000)
    config["training"].setdefault("batch_size", 18)
    config["training"].setdefault("optimizer", "AdamW")
    config["training"].setdefault("lr_scheduler", "Cosine")
    config["training"].setdefault("gpu", "NVIDIA H100")
    config["training"].setdefault("training_time", "12 hours")

    config.setdefault("model", {})
    config["model"].setdefault("point_cloud", {})
    config["model"]["point_cloud"].setdefault("num_points", 256)
    config["model"]["point_cloud"].setdefault("noise", {"probability": 0.5, "std": 0.01})
    config["model"]["point_cloud"].setdefault("embedding_dimension", 1536)
    config["model"].setdefault("llm", {})
    config["model"]["llm"].setdefault("model_name", "Qwen2-1.5B")
    config["model"].setdefault("decoder", "Auto-regressive Python code generator")

    config.setdefault("dataset", {})
    config["dataset"].setdefault("procedurally_generated_samples", 1000000)
    config["dataset"].setdefault("alternative_dataset", "DeepCAD 160k")
    config["dataset"].setdefault("cad_library", "CadQuery")

    config.setdefault("logging", {})
    config["logging"].setdefault("log_interval", 100)

    return config

# -----------------------------------------------------------------------------
# Furthest Point Sampling (FPS)
# -----------------------------------------------------------------------------
def furthest_point_sampling(point_cloud: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Downsample a dense point cloud to a fixed number of points using furthest point sampling.

    Args:
        point_cloud (np.ndarray): Input dense point cloud with shape (N, 3).
        num_samples (int): Number of points to sample.

    Returns:
        np.ndarray: Downsampled point cloud with shape (num_samples, 3).
    """
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError("Input point_cloud must be a 2D array with shape (N, 3)")

    num_total = point_cloud.shape[0]
    if num_samples >= num_total:
        return point_cloud.copy()

    sampled_indices: List[int] = []
    sampled_indices.append(0)  # Start with the first point

    # Initialize distances to infinity
    distances = np.full((num_total,), np.inf)

    # Iteratively add the furthest point from the current set
    for _ in range(1, num_samples):
        last_sampled = point_cloud[sampled_indices[-1], :]
        dist_to_last = np.sum((point_cloud - last_sampled) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        next_index = int(np.argmax(distances))
        sampled_indices.append(next_index)

    return point_cloud[sampled_indices, :]

# -----------------------------------------------------------------------------
# Fourier Positional Encoding
# -----------------------------------------------------------------------------
def fourier_encode(points: np.ndarray, num_bands: int = 6) -> np.ndarray:
    """
    Apply Fourier positional encoding to a set of 3D points.

    For each coordinate in every point, this function computes sin and cos at multiple frequencies.
    The frequencies are powers of 2 multiplied by π.

    Args:
        points (np.ndarray): Input array of shape (N, 3) representing 3D points.
        num_bands (int): Number of frequency bands to use. Defaults to 6.

    Returns:
        np.ndarray: Fourier encoded representation with shape (N, 3 * 2 * num_bands).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be a 2D array with shape (N, 3)")
    
    encoded_list = []
    for i in range(num_bands):
        frequency: float = 2 ** i * pi
        sin_enc = np.sin(points * frequency)
        cos_enc = np.cos(points * frequency)
        encoded_list.append(sin_enc)
        encoded_list.append(cos_enc)
    # Concatenate along the last dimension
    encoded_points = np.concatenate(encoded_list, axis=1)
    return encoded_points

# -----------------------------------------------------------------------------
# CAD Code Validation Utilities
# -----------------------------------------------------------------------------
def validate_python_syntax(code_str: str) -> bool:
    """
    Validate that the provided CAD code string has correct Python syntax.

    Args:
        code_str (str): The CAD code as a string.

    Returns:
        bool: True if the syntax is valid, False otherwise.
    """
    try:
        compile(code_str, "<string>", "exec")
        return True
    except SyntaxError as e:
        logging.error(f"Syntax error in CAD code: {e}")
        return False

def validate_cad_geometry(code_str: str) -> bool:
    """
    Validate the geometric correctness of the CAD code by executing it
    and verifying that a CAD model is produced without errors.

    The function attempts to execute the CAD code in a controlled environment.
    It assumes that a valid CAD model is stored in the variable "result" after execution.
    Optionally, it tries accessing result.val() to trigger computation of the underlying solid.

    Args:
        code_str (str): The CAD code as a string.

    Returns:
        bool: True if the CAD geometry is valid, False otherwise.
    """
    safe_globals = {"cq": cq, "__builtins__": {}}
    local_env = {}
    try:
        exec(code_str, safe_globals, local_env)
    except Exception as e:
        logging.error(f"Error executing CAD code: {e}")
        return False

    result = local_env.get("result", None)
    if result is None:
        logging.error("CAD code did not produce a 'result' variable.")
        return False

    # Attempt to call result.val() to verify that the CAD model can be computed.
    try:
        # Depending on the CadQuery version, result.val() returns the underlying shape.
        _ = result.val()
    except Exception as e:
        logging.error(f"CAD geometry validation failed when accessing result.val(): {e}")
        return False

    return True

def validate_cad_code(code_str: str) -> bool:
    """
    Validate a CAD code string by checking both the Python syntax and the CAD geometry validity.

    Args:
        code_str (str): The CAD code as a string.

    Returns:
        bool: True if both syntax and CAD geometry are valid, False otherwise.
    """
    if not validate_python_syntax(code_str):
        return False
    if not validate_cad_geometry(code_str):
        return False
    return True

# -----------------------------------------------------------------------------
# Logging Utility
# -----------------------------------------------------------------------------
def log_message(message: str, level: str = "info") -> None:
    """
    Log a message with the specified level using Python's logging module.

    Args:
        message (str): The message to log.
        level (str): The logging level ("debug", "info", "warning", "error", "critical").
                     Defaults to "info".
    """
    level = level.lower()
    if level == "debug":
        logging.debug(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    elif level == "critical":
        logging.critical(message)
    else:
        logging.info(message)

# -----------------------------------------------------------------------------
# Duplicate Detection Utility
# -----------------------------------------------------------------------------
def compute_md5_hash(code_str: str) -> str:
    """
    Compute an MD5 hash for a given CAD code string for duplicate detection.

    Args:
        code_str (str): The CAD code as a string.

    Returns:
        str: Hexadecimal MD5 hash of the code.
    """
    hash_object = hashlib.md5(code_str.encode("utf-8"))
    return hash_object.hexdigest()

# -----------------------------------------------------------------------------
# Sample Point Cloud Generator from CAD Code
# -----------------------------------------------------------------------------
def sample_point_cloud(code_str: str, num_points: int = 1024) -> np.ndarray:
    """
    Generate a point cloud sample from the provided CAD code.
    
    The function executes the CAD code in a controlled environment to produce a CAD model.
    It then attempts to compute the model's bounding box and uniformly samples points within it.
    If any step fails, a fallback random point cloud from a unit cube is returned.

    Args:
        code_str (str): The CAD code as a string.
        num_points (int): Number of points to sample. Defaults to 1024.

    Returns:
        np.ndarray: An array of shape (num_points, 3) representing the sampled point cloud.
    """
    safe_globals = {"cq": cq, "__builtins__": {}}
    local_env = {}
    try:
        exec(code_str, safe_globals, local_env)
        result = local_env.get("result", None)
        if result is not None:
            # Attempt to get the underlying CAD shape; CadQuery convention uses result.val()
            shape = result.val()
            # Try to compute the bounding box; use CadQuery's BoundingBox if available.
            # The BoundingBox is assumed to have attributes: xmin, xmax, ymin, ymax, zmin, zmax.
            try:
                bb = shape.BoundingBox()
                min_x, max_x = bb.xmin, bb.xmax
                min_y, max_y = bb.ymin, bb.ymax
                min_z, max_z = bb.zmin, bb.zmax
            except Exception as e:
                logging.warning(f"BoundingBox computation failed: {e}; using default bounds.")
                min_x, max_x = -1.0, 1.0
                min_y, max_y = -1.0, 1.0
                min_z, max_z = -1.0, 1.0
        else:
            logging.error("CAD code did not produce a 'result' variable. Using fallback point cloud.")
            min_x, max_x = -1.0, 1.0
            min_y, max_y = -1.0, 1.0
            min_z, max_z = -1.0, 1.0

    except Exception as e:
        logging.error(f"Error executing CAD code for point cloud sampling: {e}")
        min_x, max_x = -1.0, 1.0
        min_y, max_y = -1.0, 1.0
        min_z, max_z = -1.0, 1.0

    xs = np.random.uniform(min_x, max_x, num_points)
    ys = np.random.uniform(min_y, max_y, num_points)
    zs = np.random.uniform(min_z, max_z, num_points)
    point_cloud = np.stack([xs, ys, zs], axis=1)
    return point_cloud

# -----------------------------------------------------------------------------
# Load Dataset from File Utility
# -----------------------------------------------------------------------------
def load_dataset_from_file(dataset_name: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from a JSON file based on the provided dataset name.

    The filename is generated by replacing spaces in the dataset name with underscores
    and appending the '.json' extension.

    Args:
        dataset_name (str): The name or identifier of the dataset.

    Returns:
        List[Dict[str, Any]]: A list of dataset samples where each sample is a dictionary.
    """
    file_name = f"{dataset_name.replace(' ', '_')}.json"
    if not os.path.exists(file_name):
        logging.error(f"Dataset file {file_name} does not exist.")
        return []
    try:
        with open(file_name, "r") as file:
            data = json.load(file)
        if not isinstance(data, list):
            logging.error(f"Dataset file {file_name} does not contain a list of samples.")
            return []
        return data
    except Exception as e:
        logging.error(f"Failed to load dataset file {file_name}: {e}")
        return []
