"""model.py

This module implements the Model class for CAD-Recode that integrates
a PointCloudProcessor with a fine-tuned auto-regressive LLM decoder for
generating CAD code from 3D point clouds. The implementation follows the design
outlined in the CAD-Recode paper and uses configuration parameters from config.yaml.

Classes:
    PointCloudProcessor: Processes raw point clouds by applying furthest point sampling
                         and Fourier positional encoding.
    Model: A torch.nn.Module that projects point cloud features into a latent space and
           uses a fine-tuned LLM decoder (loaded via HuggingFace transformers) for CAD code generation.

Dependencies:
    - numpy==1.21.0
    - torch==1.9.0
    - transformers
    - cadquery
    - pythonocc-core
    - tqdm
    - utils.py (provides furthest_point_sampling and fourier_encode)

Author: Your Name
Date: Today's Date
"""

import random
from math import pi
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utility functions from utils.py
from utils import furthest_point_sampling, fourier_encode


class PointCloudProcessor:
    """Processes a 3D point cloud by applying furthest point sampling and Fourier encoding.

    Attributes:
        num_points (int): Number of points to downsample each point cloud.
        num_bands (int): Number of frequency bands for Fourier positional encoding.
    """

    def __init__(self, num_points: int = 256, num_bands: int = 6) -> None:
        """
        Initializes the PointCloudProcessor with a given number of points and Fourier bands.

        Args:
            num_points (int): Number of points to sample. Default is 256.
            num_bands (int): Number of Fourier frequency bands. Default is 6.
        """
        self.num_points: int = num_points
        self.num_bands: int = num_bands

    def process(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Processes a single point cloud sample.

        This function performs:
            1. Furthest point sampling on the input point cloud.
            2. Fourier positional encoding on the downsampled points.
        
        Args:
            point_cloud (torch.Tensor): A tensor of shape (N, 3) representing a point cloud.

        Returns:
            torch.Tensor: A tensor of shape (num_points, 3 * 2 * num_bands) with the Fourier encoded points.
        """
        # Convert the point cloud to numpy array and perform furthest point sampling.
        point_np: np.ndarray = point_cloud.detach().cpu().numpy()  # shape (N, 3)
        sampled_np: np.ndarray = furthest_point_sampling(point_np, self.num_points)  # shape (num_points, 3)
        # Apply Fourier positional encoding; output dimension = 3 * 2 * num_bands.
        encoded_np: np.ndarray = fourier_encode(sampled_np, num_bands=self.num_bands)
        # Convert to torch tensor.
        encoded_tensor: torch.Tensor = torch.tensor(encoded_np, dtype=torch.float32, device=point_cloud.device)
        return encoded_tensor

    def process_batch(self, batch_point_cloud: torch.Tensor) -> torch.Tensor:
        """Processes a batch of point clouds.

        Args:
            batch_point_cloud (torch.Tensor): Tensor of shape (B, N, 3) representing a batch.

        Returns:
            torch.Tensor: Tensor of shape (B, num_points, 3 * 2 * num_bands) with encoded representations.
        """
        batch_embeddings = []
        for i in range(batch_point_cloud.size(0)):
            sample: torch.Tensor = batch_point_cloud[i]  # shape (N, 3)
            processed_sample: torch.Tensor = self.process(sample)
            batch_embeddings.append(processed_sample)
        return torch.stack(batch_embeddings, dim=0)


class Model(nn.Module):
    """CAD-Recode Model.

    This model integrates a PointCloudProcessor for point cloud feature extraction and
    a fine-tuned auto-regressive LLM decoder (e.g., Qwen2-1.5B) for generating CAD code tokens.

    The forward() method supports both training (with teacher forcing via ground-truth code tokens)
    and inference (auto-regressive generation seeded with a start token).
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the Model using configuration parameters.

        Args:
            config (dict): Dictionary containing configuration parameters from config.yaml.
        """
        super(Model, self).__init__()

        # Retrieve point cloud configuration.
        point_cloud_config = config.get("model", {}).get("point_cloud", {})
        self.num_points: int = point_cloud_config.get("num_points", 256)
        noise_config = point_cloud_config.get("noise", {"probability": 0.5, "std": 0.01})
        self.noise_prob: float = noise_config.get("probability", 0.5)
        self.noise_std: float = noise_config.get("std", 0.01)
        self.embedding_dim: int = point_cloud_config.get("embedding_dimension", 1536)

        # Instantiate the PointCloudProcessor.
        self.pc_processor = PointCloudProcessor(num_points=self.num_points, num_bands=6)
        # Fourrier encoding output dimension is 3 * 2 * num_bands.
        self.input_feature_dim: int = 3 * 2 * 6  # default: 36

        # Linear projection layer: projects Fourier encoded features to the embedding_dim.
        self.proj_linear: nn.Linear = nn.Linear(self.input_feature_dim, self.embedding_dim)

        # Load the pre-trained LLM decoder and its tokenizer.
        llm_config = config.get("model", {}).get("llm", {})
        model_name: str = llm_config.get("model_name", "Qwen2-1.5B")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_decoder = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(
        self, point_cloud: torch.Tensor, code_input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Performs a forward pass of the model.

        The function first injects Gaussian noise to the input point cloud (with a given probability),
        processes the point cloud using the PointCloudProcessor, and applies a linear projection.
        Then, if ground-truth code tokens are provided (teacher forcing), it concatenates the projected
        point cloud tokens with the code embeddings and passes them to the LLM decoder to compute
        output logits and loss. Otherwise, it calls the generate() method to perform inference.

        Args:
            point_cloud (torch.Tensor): A tensor of shape (B, N, 3) representing a batch of point clouds.
            code_input_ids (Optional[torch.Tensor]): Ground-truth token IDs for teacher forcing,
                with shape (B, L). If None, the model performs inference.

        Returns:
            torch.Tensor: The output from the LLM decoder. In training mode, includes logits and loss.
        """
        batch_size: int = point_cloud.size(0)
        processed_list = []
        # Process each sample in the batch.
        for i in range(batch_size):
            sample = point_cloud[i]  # shape (N, 3)
            # Apply noise injection with probability self.noise_prob.
            if random.random() < self.noise_prob:
                noise = torch.randn_like(sample) * self.noise_std
                sample = sample + noise
            # Process the sample using the PointCloudProcessor.
            encoded_sample = self.pc_processor.process(sample)  # shape (num_points, encoded_dim)
            processed_list.append(encoded_sample)
        # Stack processed samples into a batch tensor.
        encoded_batch: torch.Tensor = torch.stack(processed_list, dim=0)  # shape (B, num_points, encoded_dim)

        # Apply the linear projection to map to LLM embedding space.
        proj_tokens: torch.Tensor = self.proj_linear(encoded_batch)  # shape (B, num_points, embedding_dim)

        if code_input_ids is not None:
            # Training mode with teacher forcing.
            # Retrieve code token embeddings from the LLM's embedding layer.
            code_embeddings: torch.Tensor = self.llm_decoder.get_input_embeddings()(code_input_ids)
            # Concatenate the point cloud tokens with the code token embeddings.
            inputs_embeds: torch.Tensor = torch.cat([proj_tokens, code_embeddings], dim=1)
            # Create attention mask of ones.
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=proj_tokens.device)
            # Create labels: for the prefix tokens corresponding to the point cloud, use -100 to ignore loss.
            prefix_labels = torch.full((batch_size, proj_tokens.size(1)), -100, dtype=torch.long, device=proj_tokens.device)
            # Concatenate the ignore labels with the ground truth code token IDs.
            labels = torch.cat([prefix_labels, code_input_ids], dim=1)
            # Forward pass through the LLM decoder.
            outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            return outputs
        else:
            # Inference mode: use auto-regressive generation.
            return self.generate(point_cloud)

    def generate(
        self, point_cloud: torch.Tensor, max_length: int = 256, num_return_sequences: int = 1
    ) -> torch.Tensor:
        """Generates CAD code tokens from an input point cloud using autoregressive decoding.

        This method processes the input point cloud, obtains the projected tokens, obtains the
        start token embedding from the tokenizer, concatenates them, and calls the LLM decoder's
        generate() method.

        Args:
            point_cloud (torch.Tensor): A tensor of shape (B, N, 3) representing a batch of point clouds.
            max_length (int): Maximum number of tokens to generate (excluding the point cloud tokens). Default is 256.
            num_return_sequences (int): Number of distinct generated sequences per input sample.

        Returns:
            torch.Tensor: Generated token IDs with shape (B * num_return_sequences, L_generated).
        """
        batch_size: int = point_cloud.size(0)
        processed_list = []
        # Process each sample in the batch with noise injection.
        for i in range(batch_size):
            sample = point_cloud[i]
            if random.random() < self.noise_prob:
                noise = torch.randn_like(sample) * self.noise_std
                sample = sample + noise
            encoded_sample = self.pc_processor.process(sample)
            processed_list.append(encoded_sample)
        # Stack processed samples.
        encoded_batch: torch.Tensor = torch.stack(processed_list, dim=0)
        # Linear projection.
        proj_tokens: torch.Tensor = self.proj_linear(encoded_batch)  # shape (B, num_points, embedding_dim)
        # Obtain the start token ID (assumed to be the beginning-of-sequence token, e.g., "<s>").
        start_token_id: int = self.tokenizer.bos_token_id
        if start_token_id is None:
            # Fallback: try converting "<s>" to its token ID.
            start_token_id = self.tokenizer.convert_tokens_to_ids("<s>")
        # Create a tensor of start tokens for each sample.
        start_tokens = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=proj_tokens.device)
        # Get embeddings of the start tokens.
        start_embeddings = self.llm_decoder.get_input_embeddings()(start_tokens)  # shape (B, 1, embedding_dim)
        # Concatenate the projected point cloud tokens with the start token embeddings.
        inputs_embeds = torch.cat([proj_tokens, start_embeddings], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=proj_tokens.device)
        # Use the LLM decoder's generate() method with inputs_embeds.
        generated_tokens: torch.Tensor = self.llm_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=proj_tokens.size(1) + max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
        )
        return generated_tokens
