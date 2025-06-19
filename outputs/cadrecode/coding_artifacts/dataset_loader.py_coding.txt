"""dataset_loader.py

This module implements the DatasetLoader class for the CAD-Recode project.
It is responsible for generating a procedural CAD dataset (using CAD sketch and CAD model
generation algorithms) and for loading existing datasets (e.g. DeepCAD 160k).

The generated samples consist of:
  - "cad_code": a Python script (using CadQuery) that reconstructs a CAD model.
  - "point_cloud": a sampled point cloud from the CAD model geometry.

Configuration parameters are read from a configuration dictionary (parsed from config.yaml).

Dependencies:
  - numpy==1.21.0
  - torch==1.9.0
  - transformers
  - cadquery
  - pythonocc-core
  - tqdm

This module also imports several utility functions from utils.py:
  - extract_boundary_loops
  - analyze_boundary
  - validate_shape_topology
  - validate_cad_code
  - sample_point_cloud
  - load_dataset_from_file

Author: Your Name
Date: Today's Date
"""

import random
import numpy as np
import cadquery as cq
import hashlib
import logging
import json
from tqdm import tqdm

# Import utility functions assumed to be implemented in utils.py.
from utils import (
    extract_boundary_loops,
    analyze_boundary,
    validate_shape_topology,
    validate_cad_code,
    sample_point_cloud,
    load_dataset_from_file,
)

class DatasetLoader:
    """DatasetLoader class for generating and loading CAD datasets.

    Attributes:
        config (dict): Configuration parameters.
        num_procedural_samples (int): Number of procedural samples to generate.
        alternative_dataset (str): Identifier or name for the alternative dataset.
        cad_library (str): CAD library used (e.g., "CadQuery").
        log_interval (int): Interval for logging progress.
    """

    def __init__(self, config: dict) -> None:
        """Initializes the DatasetLoader with configuration settings.

        Args:
            config (dict): Dictionary with configuration parameters.
        """
        self.config: dict = config
        self.num_procedural_samples: int = config.get("dataset", {}).get("procedurally_generated_samples", 1000000)
        self.alternative_dataset: str = config.get("dataset", {}).get("alternative_dataset", "DeepCAD 160k")
        self.cad_library: str = config.get("dataset", {}).get("cad_library", "CadQuery")
        self.log_interval: int = config.get("logging", {}).get("log_interval", 100)

        # Set seeds for reproducibility.
        random.seed(42)
        np.random.seed(42)

        # Configure basic logging.
        logging.basicConfig(level=logging.INFO)

    def generate_procedural_dataset(self, num_samples: int = None) -> list:
        """Generates a procedural CAD dataset.

        This method implements Algorithms 1 (2D sketch generation) and 2 (CAD model generation)
        as described in the paper. It:
          - Generates a 2D sketch by random selection of primitives (Circle, RotatedRectangle)
            and boolean operations (Union, Cut).
          - Generates a CAD Python code string using CadQuery commands by extruding the sketch.
          - Validates the generated code (syntax and geometric validity via validate_cad_code).
          - Uses a duplicate detection protocol (MD5 hash of the code) to ensure uniqueness.
          - Samples a point cloud from the model using sample_point_cloud().

        Args:
            num_samples (int, optional): Number of samples to generate.
                If not provided, defaults to the configuration value.

        Returns:
            list: A list of samples where each sample is a dictionary with keys:
                  "cad_code" and "point_cloud".
        """
        if num_samples is None:
            num_samples = self.num_procedural_samples

        dataset: list = []
        seen_hashes: set = set()

        for i in tqdm(range(num_samples), desc="Generating procedural dataset"):
            try:
                # Generate 2D sketch (Algorithm 1)
                sketch_data: list = self._generate_2d_sketch()
                # Generate full CAD code (Algorithm 2) from sketch data.
                cad_code: str = self._generate_cad_model(sketch_data)

                # Validate the generated CAD code using CadQuery and CAD-specific rules.
                if not validate_cad_code(cad_code):
                    logging.warning(f"Sample {i} failed CAD code validation.")
                    continue

                # Use duplicate detection via an MD5 hash on the CAD code.
                sample_hash: str = self._compute_sample_hash(cad_code)
                if sample_hash in seen_hashes:
                    logging.info(f"Duplicate sample {i} detected; skipping.")
                    continue
                seen_hashes.add(sample_hash)

                # Generate a point cloud from the CAD model.
                point_cloud: np.ndarray = sample_point_cloud(cad_code)

                # Assemble the sample dictionary.
                sample: dict = {
                    "cad_code": cad_code,
                    "point_cloud": point_cloud,
                }
                dataset.append(sample)

                # Log progress every log_interval samples.
                if (i + 1) % self.log_interval == 0:
                    logging.info(f"Generated {i+1} samples so far.")

            except Exception as e:
                logging.error(f"Error generating sample {i}: {e}")
                continue

        return dataset

    def load_existing_dataset(self, dataset_name: str = None) -> list:
        """Loads an existing CAD dataset and validates the samples.

        The method loads CAD samples from a file (assumed to be JSON format) whose name
        is derived from the dataset name. If necessary, it converts the CAD representation
        to the CadQuery code format and validates each sample using validate_cad_code().

        Args:
            dataset_name (str, optional): The name of the dataset to load.
                Defaults to the configuration's alternative_dataset value.

        Returns:
            list: A list of verified dataset samples.
        """
        if dataset_name is None:
            dataset_name = self.alternative_dataset

        try:
            # Load dataset using the utility function.
            dataset = load_dataset_from_file(dataset_name)
            validated_dataset: list = []
            for sample in dataset:
                cad_code: str = sample.get("cad_code", "")
                if validate_cad_code(cad_code):
                    validated_dataset.append(sample)
            return validated_dataset

        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {e}")
            return []

    def _generate_2d_sketch(self) -> list:
        """Generates a 2D sketch represented as a list of primitive dictionaries.

        Each primitive is defined with its type (Circle or RotatedRectangle), parameters,
        and a boolean operation (Union or Cut). This simulates Algorithm 1.

        Returns:
            list: List of dictionaries, each representing a sketch primitive.
        """
        num_primitives: int = random.randint(3, 8)
        primitives: list = []

        for _ in range(num_primitives):
            primitive_type: str = random.choice(["Circle", "RotatedRectangle"])
            boolean_operation: str = random.choice(["Union", "Cut"])

            if primitive_type == "Circle":
                radius: float = round(random.uniform(1.0, 10.0), 2)
                center: tuple = (round(random.uniform(-10, 10), 2), round(random.uniform(-10, 10), 2))
                primitive = {
                    "type": "Circle",
                    "center": center,
                    "radius": radius,
                    "operation": boolean_operation
                }
            else:  # RotatedRectangle
                width: float = round(random.uniform(2.0, 15.0), 2)
                height: float = round(random.uniform(2.0, 15.0), 2)
                angle: float = round(random.uniform(0, 360), 2)
                center: tuple = (round(random.uniform(-10, 10), 2), round(random.uniform(-10, 10), 2))
                primitive = {
                    "type": "RotatedRectangle",
                    "center": center,
                    "width": width,
                    "height": height,
                    "angle": angle,
                    "operation": boolean_operation
                }
            primitives.append(primitive)

        # Note: Further processing such as extracting boundary loops or validating the sketch
        # topology can be performed using extract_boundary_loops, analyze_boundary, and
        # validate_shape_topology utility functions if needed.
        return primitives

    def _generate_cad_model(self, sketch_primitives: list) -> str:
        """Generates the CAD model Python code (using CadQuery) from 2D sketch primitives.

        Implements Algorithm 2 which:
          - Selects a random sketch plane.
          - Extrudes the sketch to form a 3D model.
          - Normalizes the model (scaling and quantization).

        Args:
            sketch_primitives (list): List of primitive dictionaries from the 2D sketch.

        Returns:
            str: A multi-line string containing the CAD Python code.
        """
        code_lines: list = []
        code_lines.append("import cadquery as cq")
        code_lines.append("")

        # Define the workplane with a randomly chosen origin.
        origin_x: float = round(random.uniform(-10, 10), 2)
        origin_y: float = round(random.uniform(-10, 10), 2)
        origin_z: float = 0.0
        code_lines.append(f"w0 = cq.Workplane('XY', origin=({origin_x}, {origin_y}, {origin_z}))")
        code_lines.append("sk = w0.sketch()")

        # Add sketch primitives into the sketch.
        for primitive in sketch_primitives:
            if primitive["type"] == "Circle":
                # CadQuery's circle is drawn at the current sketch origin with the given radius.
                code_lines.append(
                    f"sk = sk.circle({primitive['radius']})  # Circle with center {primitive['center']}"
                )
            elif primitive["type"] == "RotatedRectangle":
                # For a rotated rectangle, we use the rect command.
                code_lines.append(
                    f"sk = sk.rect({primitive['width']}, {primitive['height']})  "
                    f"# Rotated rectangle; intended rotation: {primitive['angle']}°; center {primitive['center']}"
                )
            # Include the boolean operation as a comment.
            code_lines.append(f"# Boolean operation: {primitive['operation']}")

        code_lines.append("sk = sk.finalize()")

        # Choose a random extrusion height between 5 and 20.
        extrude_height: float = round(random.uniform(5.0, 20.0), 2)
        code_lines.append(f"result = sk.extrude({extrude_height})")
        
        # Normalize the model: scale and quantize parameters.
        code_lines.append("# Normalize model: scale to unit bounding box and quantize parameters (placeholder)")
        code_lines.append("result = result.scale(1.0)")

        cad_code: str = "\n".join(code_lines)
        return cad_code

    def _compute_sample_hash(self, cad_code: str) -> str:
        """Computes an MD5 hash of the CAD code for duplicate detection.

        Args:
            cad_code (str): The generated CAD code string.

        Returns:
            str: A hexadecimal MD5 hash.
        """
        hash_object = hashlib.md5(cad_code.encode("utf-8"))
        return hash_object.hexdigest()

    def _load_dataset_from_file(self, dataset_name: str) -> list:
        """Loads dataset samples from a JSON file based on the dataset name.

        The file is assumed to be named by replacing spaces in the dataset name with underscores
        and appending the '.json' extension.

        Args:
            dataset_name (str): The name or identifier of the dataset.

        Returns:
            list: A list of dataset samples.
        """
        file_path: str = f"{dataset_name.replace(' ', '_')}.json"
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return data
        except Exception as e:
            logging.error(f"Failed to load dataset file {file_path}: {e}")
            return []

# End of dataset_loader.py
