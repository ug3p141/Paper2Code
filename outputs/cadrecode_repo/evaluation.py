"""evaluation.py

This module implements the Evaluation class for the CAD-Recode project.
It evaluates the performance of the trained CAD-Recode model by comparing generated
CAD code outputs (CadQuery Python code) against ground-truth CAD models using the following metrics:
  - Chamfer Distance (CD): Difference between predicted and ground-truth point clouds.
  - Intersection over Union (IoU): Overlap between voxelized representations of the CAD models.
  - Invalidity Ratio (IR): Ratio of samples for which the model did not produce a valid CAD model.

The evaluation process performs test-time sampling by generating multiple candidate CAD code 
predictions for each input point cloud and selecting the best valid candidate based on the lowest CD.

Dependencies:
    - numpy==1.21.0
    - torch==1.9.0
    - transformers
    - cadquery
    - pythonocc-core
    - tqdm

Author: Your Name
Date: Today's Date
"""

import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

# Import utility functions from utils.py.
from utils import validate_cad_code, sample_point_cloud

# Set up logger
logging.basicConfig(level=logging.INFO)


class Evaluation:
    """
    Evaluation class for CAD-Recode.
    
    Attributes:
        model (torch.nn.Module): The trained CAD-Recode model.
        dataset (List[Dict[str, Any]]): List of samples. Each sample should have at least:
            - "point_cloud": numpy.ndarray (N, 3) of the input point cloud.
            - "cad_code": string of the ground-truth CAD Python code.
        config (dict): Configuration dictionary. Uses eval-specific settings if provided.
        num_candidates (int): Number of test-time sampling candidates. Default is 10.
        log_interval (int): Logging interval for progress reporting.
        device (torch.device): The device on which the model is located.
    """
    def __init__(self, model: torch.nn.Module, dataset: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the Evaluation instance.
        
        Args:
            model (torch.nn.Module): The trained CAD-Recode model.
            dataset (List[Dict[str, Any]]): Dataset samples.
            config (Optional[Dict[str, Any]]): Configuration parameters; if None, defaults are used.
        """
        self.model = model
        self.dataset = dataset
        self.config = config if config is not None else {}
        # Set default number of test-time candidates; override if provided in config under "evaluation".
        self.num_candidates: int = self.config.get("evaluation", {}).get("num_candidates", 10)
        self.log_interval: int = self.config.get("logging", {}).get("log_interval", 100)
        self.device: torch.device = next(self.model.parameters()).device if next(self.model.parameters(), None) is not None else torch.device("cpu")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the trained model on the provided dataset.
        
        For each sample, the method:
          - Generates candidate CAD code outputs via test-time sampling.
          - Executes and validates each candidate CAD code using CadQuery.
          - For valid predictions, samples 8192 points from both the predicted and
            ground-truth CAD models.
          - Computes the Chamfer Distance and IoU between the predicted and ground-truth point clouds.
          - Selects the candidate with the minimum Chamfer Distance if any valid candidate exists.
          - Otherwise, marks the sample as invalid.
        
        Returns:
            Dict[str, Any]: A dictionary containing the aggregated metrics:
                - "mean_CD": Mean Chamfer Distance over valid samples.
                - "median_CD": Median Chamfer Distance over valid samples.
                - "IoU": Average Intersection over Union (in percentage) over valid samples.
                - "IR": Invalidity Ratio (fraction of samples with no valid prediction).
                - "total_samples": Total number of samples processed.
                - "valid_samples": Number of samples with at least one valid candidate.
        """
        chamfer_distances: List[float] = []
        iou_values: List[float] = []
        invalid_count: int = 0
        total_samples: int = len(self.dataset)
        
        # Set model to evaluation mode.
        self.model.eval()
        
        logging.info("Starting evaluation on %d samples.", total_samples)
        
        for idx, sample in enumerate(tqdm(self.dataset, desc="Evaluating", unit="sample")):
            try:
                # Extract input point cloud and ground-truth CAD code.
                pc_np = sample.get("point_cloud")
                gt_cad_code: str = sample.get("cad_code", "")
                if pc_np is None or not gt_cad_code:
                    logging.warning("Sample %d is missing required fields.", idx)
                    invalid_count += 1
                    continue
                
                # Convert point cloud to torch tensor and move to device.
                point_cloud_tensor: torch.Tensor = torch.tensor(pc_np, dtype=torch.float32, device=self.device)
                point_cloud_tensor = point_cloud_tensor.unsqueeze(0)  # shape (1, N, 3)

                # Execute the ground-truth CAD code and sample 8192 points from the ground-truth CAD model.
                try:
                    gt_point_cloud: np.ndarray = sample_point_cloud(gt_cad_code, num_points=8192)
                except Exception as e:
                    logging.error("Failed to sample ground-truth point cloud for sample %d: %s", idx, str(e))
                    invalid_count += 1
                    continue

                # Generate candidate predictions (test-time sampling).
                with torch.no_grad():
                    candidate_ids: torch.Tensor = self.model.generate(
                        point_cloud_tensor, max_length=256, num_return_sequences=self.num_candidates
                    )
                # candidate_ids has shape (num_candidates, L_generated)
                candidate_codes: List[str] = []
                for i in range(candidate_ids.size(0)):
                    # Decode each candidate using the model's tokenizer.
                    token_ids = candidate_ids[i].tolist()
                    candidate_code: str = self.model.tokenizer.decode(token_ids, skip_special_tokens=True)
                    candidate_codes.append(candidate_code)
                
                valid_candidates: List[Tuple[str, np.ndarray, float, float]] = []
                # Evaluate each candidate.
                for cand_code in candidate_codes:
                    # Validate CAD code (syntax and geometry).
                    if not validate_cad_code(cand_code):
                        continue
                    try:
                        # Sample 8192 points from the candidate CAD model.
                        cand_point_cloud: np.ndarray = sample_point_cloud(cand_code, num_points=8192)
                    except Exception as exec_err:
                        logging.error("Error sampling candidate CAD code: %s", str(exec_err))
                        continue
                    # Compute Chamfer Distance.
                    cd_value: float = self.compute_chamfer_distance(cand_point_cloud, gt_point_cloud)
                    # Compute IoU.
                    iou_value: float = self.compute_iou(cand_point_cloud, gt_point_cloud)
                    valid_candidates.append((cand_code, cand_point_cloud, cd_value, iou_value))
                
                if len(valid_candidates) == 0:
                    # No valid candidate generated for this sample.
                    invalid_count += 1
                    continue
                
                # Select the candidate with the minimum Chamfer Distance.
                best_candidate = min(valid_candidates, key=lambda item: item[2])
                best_cd: float = best_candidate[2]
                best_iou: float = best_candidate[3]
                
                chamfer_distances.append(best_cd)
                iou_values.append(best_iou)
            
            except Exception as e:
                logging.error("Error evaluating sample %d: %s", idx, str(e))
                invalid_count += 1
                continue
            
            if (idx + 1) % self.log_interval == 0:
                logging.info("Processed %d/%d samples.", idx + 1, total_samples)
        
        num_valid: int = total_samples - invalid_count
        # Aggregate metrics.
        if chamfer_distances:
            mean_cd: float = float(np.mean(chamfer_distances))
            median_cd: float = float(np.median(chamfer_distances))
        else:
            mean_cd = float('inf')
            median_cd = float('inf')
        if iou_values:
            mean_iou: float = float(np.mean(iou_values))
        else:
            mean_iou = 0.0
        
        invalidity_ratio: float = self.compute_invalidity_ratio(total_samples, invalid_count)
        
        metrics: Dict[str, Any] = {
            "mean_CD": mean_cd,
            "median_CD": median_cd,
            "IoU": mean_iou,
            "IR": invalidity_ratio,
            "total_samples": total_samples,
            "valid_samples": num_valid,
        }
        
        logging.info("Evaluation complete. Mean CD: %.4f, Median CD: %.4f, Mean IoU: %.2f%%, IR: %.2f%%",
                     mean_cd, median_cd, mean_iou * 100, invalidity_ratio * 100)
        
        return metrics

    @staticmethod
    def compute_chamfer_distance(pred_points: np.ndarray, target_points: np.ndarray) -> float:
        """
        Computes the Chamfer Distance between two point clouds.

        For each point in pred_points, finds the closest point in target_points and vice versa.
        The Chamfer Distance is the average of these minimum distances.

        Args:
            pred_points (np.ndarray): Predicted point cloud as an array of shape (N, 3).
            target_points (np.ndarray): Ground-truth point cloud as an array of shape (M, 3).

        Returns:
            float: The Chamfer Distance.
        """
        if pred_points.size == 0 or target_points.size == 0:
            return float('inf')
        
        # Compute pairwise squared distances.
        diff = np.expand_dims(pred_points, axis=1) - np.expand_dims(target_points, axis=0)
        dist_squared = np.sum(diff ** 2, axis=2)  # Shape (N, M)
        # For each point in pred_points, the minimum squared distance in target_points.
        min_dist_pred = np.min(dist_squared, axis=1)
        # For each point in target_points, the minimum squared distance in pred_points.
        min_dist_target = np.min(dist_squared, axis=0)
        # Compute mean of square roots.
        chamfer = np.mean(np.sqrt(min_dist_pred)) + np.mean(np.sqrt(min_dist_target))
        return chamfer

    @staticmethod
    def compute_iou(pred_points: np.ndarray, target_points: np.ndarray, grid_resolution: int = 32) -> float:
        """
        Computes the Intersection over Union (IoU) between two point clouds via voxelization.
        
        The point clouds are voxelized into a binary occupancy grid and IoU is computed as:
            IoU = (Intersection voxels) / (Union voxels)
        
        Args:
            pred_points (np.ndarray): Predicted point cloud of shape (N, 3).
            target_points (np.ndarray): Ground-truth point cloud of shape (M, 3).
            grid_resolution (int): Number of voxels per axis. Defaults to 32.
        
        Returns:
            float: IoU value (between 0 and 1).
        """
        # Determine the combined bounding box.
        all_points = np.concatenate([pred_points, target_points], axis=0)
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        # Avoid division by zero.
        bbox_range = np.maximum(max_coords - min_coords, 1e-6)
        
        # Compute voxel indices for predicted points.
        pred_indices = np.floor((pred_points - min_coords) / bbox_range * grid_resolution).astype(np.int32)
        target_indices = np.floor((target_points - min_coords) / bbox_range * grid_resolution).astype(np.int32)
        
        # Clip indices to be within [0, grid_resolution-1]
        np.clip(pred_indices, 0, grid_resolution - 1, out=pred_indices)
        np.clip(target_indices, 0, grid_resolution - 1, out=target_indices)
        
        # Convert indices to tuples and create sets.
        pred_voxels = {tuple(idx) for idx in pred_indices}
        target_voxels = {tuple(idx) for idx in target_indices}
        
        intersection = pred_voxels.intersection(target_voxels)
        union = pred_voxels.union(target_voxels)
        
        if len(union) == 0:
            return 0.0
        iou = len(intersection) / len(union)
        return iou

    @staticmethod
    def compute_invalidity_ratio(total_samples: int, invalid_samples: int) -> float:
        """
        Computes the Invalidity Ratio (IR) as the fraction of samples for which the model
        failed to produce a valid CAD model.
        
        Args:
            total_samples (int): The total number of samples evaluated.
            invalid_samples (int): The number of samples with invalid predictions.
        
        Returns:
            float: Invalidity Ratio.
        """
        if total_samples == 0:
            return 0.0
        return invalid_samples / total_samples

# End of evaluation.py
