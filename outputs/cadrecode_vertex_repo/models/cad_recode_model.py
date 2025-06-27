```python
"""
Main CAD-Recode model implementation.
Based on paper: "CAD-Recode: Reverse Engineering CAD Code from Point Clouds"

This module implements the core CAD-Recode architecture from Figure 4 of the paper,
combining a lightweight point cloud projector with a pre-trained LLM (Qwen2-1.5B)
to translate 3D point clouds into executable CadQuery Python code.

The model implements the mapping: Ψ: R^(n×3) → C
where C is the set of valid CAD codes as defined in Section 4.1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
import time
import gc
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        GenerationConfig,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Model functionality will be limited.")

# Import project modules
from models.point_cloud_projector import PointCloudProjector
from utils.cad_validation import CADValidator


# Set up logging
logger = logging.getLogger(__name__)


class CADRecodeModel(nn.Module):
    """
    Main CAD-Recode model implementing the architecture from Figure 4.
    
    Combines a point cloud projector (Ψ_p) with a pre-trained LLM decoder (Ψ_LLM)
    to map point clouds to valid CAD code. The model processes concatenated tokens
    [Q_p; Q_t] where Q_p are point cloud query tokens and Q_t are text tokens.
    
    Architecture:
    1. Point Cloud Projector: P ∈ R^(n×3) → Q_p ∈ R^(n_p×d_q)
    2. Token Concatenation: [Q_p; Q_t] ∈ R^((n_p+n_t)×d_q)
    3. LLM Decoder: Autoregressive generation of CAD code tokens
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CAD-Recode model with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If model loading fails
        """
        super(CADRecodeModel, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library is required but not available")
        
        # Store configuration
        self.config = config
        
        # Extract model configuration
        model_config = config.get('model', {})
        self.llm_model_name = model_config.get('llm_model_name', 'Qwen/Qwen2-1.5B')
        self.embedding_dim = model_config.get('embedding_dim', 1536)
        self.num_points = model_config.get('num_points', 256)
        self.start_token = model_config.get('start_token', '<s>')
        self.end_token = model_config.get('end_token', '<e>')
        
        # Extract training configuration
        training_config = config.get('training', {})
        self.training_mode = training_config.get('training_mode', 'end_to_end')
        
        # Extract system configuration
        system_config = config.get('system', {})
        self.device = system_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._initialize_tokenizer()
        self._initialize_point_projector()
        self._initialize_llm()
        self._initialize_validator()
        
        # Set training mode configuration
        self._configure_training_mode()
        
        logger.info(f"CADRecodeModel initialized:")
        logger.info(f"  LLM: {self.llm_model_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Num points: {self.num_points}")
        logger.info(f"  Training mode: {self.training_mode}")
        logger.info(f"  Device: {self.device}")
    
    def _initialize_tokenizer(self) -> None:
        """Initialize HuggingFace tokenizer with special tokens."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Handle tokenizers without pad token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Add special tokens for CAD code boundaries
            special_tokens_dict = {
                'additional_special_tokens': [self.start_token, self.end_token]
            }
            
            # Only add if tokens don't already exist
            existing_tokens = set(self.tokenizer.get_vocab().keys())
            new_tokens = [token for token in special_tokens_dict['additional_special_tokens'] 
                         if token not in existing_tokens]
            
            if new_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
                logger.info(f"Added special tokens: {new_tokens}")
            
            # Store token IDs for efficient access
            self.start_token_id = self.tokenizer.convert_tokens_to_ids(self.start_token)
            self.end_token_id = self.tokenizer.convert_tokens_to_ids(self.end_token)
            self.pad_token_id = self.tokenizer.pad_token_id
            
            logger.info(f"Tokenizer initialized with vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise RuntimeError(f"Tokenizer initialization failed: {e}")
    
    def _initialize_point_projector(self) -> None:
        """Initialize point cloud projector component."""
        try:
            self.point_projector = PointCloudProjector(
                num_points=self.num_points,
                embed_dim=self.embedding_dim,
                config=self.config
            )
            
            logger.info(f"Point projector initialized: {self.num_points} points → {self.embedding_dim}D")
            
        except Exception as e:
            logger.error(f"Failed to initialize point projector: {e}")
            raise RuntimeError(f"Point projector initialization failed: {e}")
    
    def _initialize_llm(self) -> None:
        """Initialize pre-trained LLM decoder."""
        try:
            # Load pre-trained model
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for stable training
                device_map=None  # Handle device placement manually
            )
            
            # Resize token embeddings if we added special tokens
            if len(self.tokenizer) > self.llm.config.vocab_size:
                self.llm.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Resized LLM embeddings to {len(self.tokenizer)} tokens")
            
            # Verify embedding dimension compatibility
            llm_hidden_size = self.llm.config.hidden_size
            if llm_hidden_size != self.embedding_dim:
                logger.warning(f"LLM hidden size ({llm_hidden_size}) != embedding_dim ({self.embedding_dim})")
                # Create projection layer if dimensions don't match
                self.embedding_projection = nn.Linear(self.embedding_dim, llm_hidden_size)
                logger.info(f"Added embedding projection: {self.embedding_dim} → {llm_hidden_size}")
            else:
                self.embedding_projection = None
            
            logger.info(f"LLM initialized: {self.llm_model_name}")
            logger.info(f"  Hidden size: {llm_hidden_size}")
            logger.info(f"  Vocab size: {self.llm.config.vocab_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")
    
    def _initialize_validator(self) -> None:
        """Initialize CAD code validator."""
        try:
            cadquery_config = self.config.get('cadquery', {})
            self.validator = CADValidator(cadquery_config)
            logger.info("CAD validator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize validator: {e}")
            # Continue without validator - validation can be done externally
            self.validator = None
            logger.warning("Continuing without CAD validator")
    
    def _configure_training_mode(self) -> None:
        """Configure training mode according to paper specifications."""
        if self.training_mode == 'end_to_end':
            # Point cloud projector trains from scratch
            for param in self.point_projector.parameters():
                param.requires_grad = True
            
            # LLM fine-tunes from pre-trained weights
            for param in self.llm.parameters():
                param.requires_grad = True
            
            logger.info("Configured for end-to-end training")
            
        elif self.training_mode == 'freeze_llm':
            # Only train point cloud projector
            for param in self.point_projector.parameters():
                param.requires_grad = True
            
            for param in self.llm.parameters():
                param.requires_grad = False
            
            logger.info("Configured with frozen LLM")
            
        elif self.training_mode == 'freeze_projector':
            # Only fine-tune LLM
            for param in self.point_projector.parameters():
                param.requires_grad = False
            
            for param in self.llm.parameters():
                param.requires_grad = True
            
            logger.info("Configured with frozen point projector")
        
        else:
            logger.warning(f"Unknown training mode: {self.training_mode}")
    
    def forward(
        self, 
        point_cloud: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass implementing core CAD-Recode architecture.
        
        Processes concatenated point cloud and text tokens through the LLM
        for next-token prediction as described in Section 4.2.
        
        Args:
            point_cloud: Input point cloud tensor of shape (batch_size, num_points, 3)
            input_ids: Text token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask for text tokens (batch_size, seq_len)
            labels: Target labels for training (batch_size, seq_len)
            return_dict: Whether to return dictionary or tensor
            
        Returns:
            If return_dict=True: Dictionary with 'logits', 'loss' (if labels provided)
            If return_dict=False: Logits tensor or (logits, loss) tuple
            
        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If forward pass fails
        """
        # Input validation
        if not isinstance(point_cloud, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for point_cloud, got {type(point_cloud)}")
        
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor for input_ids, got {type(input_ids)}")
        
        if point_cloud.dim() != 3 or point_cloud.size(-1) != 3:
            raise ValueError(f"Expected point_cloud shape (batch_size, num_points, 3), got {point_cloud.shape}")
        
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids shape (batch_size, seq_len), got {input_ids.shape}")
        
        batch_size = point_cloud.size(0)
        if input_ids.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: point_cloud {batch_size}, input_ids {input_ids.size(0)}")
        
        try:
            # Step 1: Process point cloud through projector
            # P ∈ R^(batch_size × num_points × 3) → Q_p ∈ R^(batch_size × num_points × embed_dim)
            point_query_tokens = self.point_projector(point_cloud)
            
            # Apply embedding projection if needed
            if self.embedding_projection is not None:
                point_query_tokens = self.embedding_projection(point_query_tokens)
            
            # Step 2: Get text embeddings from LLM
            # Convert input_ids to embeddings: Q_t ∈ R^(batch_size × seq_len × hidden_size)
            text_embeddings = self.llm.get_input_embeddings()(input_ids)
            
            # Step 3: Concatenate point and text tokens
            # [Q_p; Q_t] ∈ R^(batch_size × (num_points + seq_len) × hidden_size)
            concatenated_embeddings = torch.cat([point_query_tokens, text_embeddings], dim=1)
            
            # Step 4: Create attention mask for concatenated sequence
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            
            # Point cloud tokens are always attended to
            point_attention = torch.ones(
                batch_size, self.num_points, 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            
            # Concatenate attention masks
            full_attention_mask = torch.cat([point_attention, attention_mask], dim=1)
            
            # Step 5: Process through LLM
            # Handle different LLM architectures
            if hasattr(self.llm, 'forward'):
                llm_outputs = self.llm(
                    inputs_embeds=concatenated_embeddings,
                    attention_mask=full_attention_mask,
                    labels=self._prepare_labels(labels, batch_size) if labels is not None else None,
                    return_dict=True,
                    use_cache=False  # Disable for training
                )
            else:
                raise RuntimeError("LLM does not support inputs_embeds")
            
            # Extract outputs
            logits = llm_outputs.logits
            loss = llm_outputs.loss if labels is not None else None
            
            # Only return logits for text positions (skip point cloud positions)
            text_logits = logits[:, self.num_points:, :]
            
            if return_dict:
                output_dict = {'logits': text_logits}
                if loss is not None:
                    output_dict['loss'] = loss
                return output_dict
            else:
                if loss is not None:
                    return text_logits, loss
                else:
                    return text_logits
                    
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise RuntimeError(f"Forward pass failed: {e}")
    
    def _prepare_labels(self