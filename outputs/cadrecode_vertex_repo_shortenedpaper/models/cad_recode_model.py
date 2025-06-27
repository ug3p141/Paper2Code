```python
"""
CAD-Recode Model implementation for CAD code generation from point clouds.
Integrates point cloud projector with pre-trained LLM for end-to-end CAD reverse engineering.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer,
    GenerationConfig
)

from config import Config
from models.point_cloud_projector import PointCloudProjector


class CADRecodeModel(nn.Module):
    """
    CAD-Recode model that combines point cloud projector with LLM for CAD code generation.
    Implements the complete architecture described in the paper for reverse engineering CAD code from point clouds.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2-1.5B",
                 projector_params: Optional[Dict[str, Any]] = None):
        """
        Initialize CAD-Recode model.
        
        Args:
            model_name: Pre-trained LLM model name/path
            projector_params: Parameters for point cloud projector
        """
        super().__init__()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Store configuration
        self.model_name = model_name
        self.projector_params = projector_params if projector_params is not None else {}
        
        # Initialize model components
        self._initialize_components()
        
        # Generation parameters (will be set from config)
        self.temperature = 0.7
        self.do_sample = True
        self.top_p = 0.9
        self.max_length = 512
        self.mixed_precision = True
        
        # Device tracking
        self.device = torch.device('cpu')
        
        self.logger.info(f"Initialized CADRecodeModel with {model_name}")
    
    def _initialize_components(self) -> None:
        """Initialize point cloud projector and LLM components."""
        try:
            # Initialize tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'  # For generation
            )
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize LLM
            self.llm = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.mixed_precision else torch.float32,
                device_map=None  # We'll handle device placement manually
            )
            
            # Get model dimensions
            self.hidden_dim = self.llm.config.hidden_size
            self.vocab_size = self.llm.config.vocab_size
            
            # Set default projector parameters if not provided
            default_projector_params = {
                'num_points': 256,
                'hidden_dim': self.hidden_dim,
                'fourier_freqs': 10
            }
            default_projector_params.update(self.projector_params)
            self.projector_params = default_projector_params
            
            # Initialize point cloud projector
            self.projector = PointCloudProjector(
                num_points=self.projector_params['num_points'],
                hidden_dim=self.projector_params['hidden_dim'],
                fourier_freqs=self.projector_params['fourier_freqs']
            )
            
            # Add language modeling head if not present
            if not hasattr(self.llm, 'lm_head'):
                self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
            else:
                self.lm_head = self.llm.lm_head
            
            # Store number of point cloud tokens
            self.num_pc_tokens = self.projector_params['num_points']
            
            self.logger.info(f"Model components initialized: hidden_dim={self.hidden_dim}, "
                           f"vocab_size={self.vocab_size}, num_pc_tokens={self.num_pc_tokens}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model components: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    @classmethod
    def from_config(cls, config: Config) -> 'CADRecodeModel':
        """
        Create CADRecodeModel from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            CADRecodeModel: Configured model instance
        """
        projector_params = {
            'num_points': config.model.projector.num_points,
            'hidden_dim': config.model.projector.hidden_dim,
            'fourier_freqs': config.model.projector.fourier_freqs
        }
        
        model = cls(
            model_name=config.model.llm_model_name,
            projector_params=projector_params
        )
        
        # Set generation parameters from config
        model.temperature = config.model.generation.temperature
        model.do_sample = config.model.generation.do_sample
        model.top_p = config.model.generation.top_p
        model.max_length = config.model.generation.max_length
        model.mixed_precision = getattr(config.training, 'mixed_precision', True)
        
        return model
    
    def _validate_inputs(self, 
                        point_clouds: torch.Tensor, 
                        target_codes: Optional[torch.Tensor] = None) -> None:
        """
        Validate input tensors.
        
        Args:
            point_clouds: Point cloud tensor
            target_codes: Target code tensor (optional)
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate point clouds
        if not isinstance(point_clouds, torch.Tensor):
            raise ValueError("point_clouds must be a torch.Tensor")
        
        if point_clouds.ndim != 3 or point_clouds.size(-1) != 3:
            raise ValueError(f"point_clouds must have shape [batch_size, num_points, 3], "
                           f"got {point_clouds.shape}")
        
        if point_clouds.size(0) == 0:
            raise ValueError("Batch size cannot be zero")
        
        if point_clouds.size(1) == 0:
            raise ValueError("Point cloud cannot be empty")
        
        # Check for invalid coordinates
        if not torch.isfinite(point_clouds).all():
            raise ValueError("point_clouds contains invalid coordinates (NaN/Inf)")
        
        # Validate target codes if provided
        if target_codes is not None:
            if not isinstance(target_codes, torch.Tensor):
                raise ValueError("target_codes must be a torch.Tensor")
            
            if target_codes.ndim != 2:
                raise ValueError(f"target_codes must have shape [batch_size, seq_len], "
                               f"got {target_codes.shape}")
            
            if target_codes.size(0) != point_clouds.size(0):
                raise ValueError(f"Batch size mismatch: point_clouds={point_clouds.size(0)}, "
                               f"target_codes={target_codes.size(0)}")
    
    def _create_attention_mask(self, 
                              point_clouds: torch.Tensor, 
                              target_codes: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for combined point cloud and text tokens.
        
        Args:
            point_clouds: Point cloud tensor [batch_size, num_points, 3]
            target_codes: Target code tensor [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Attention mask [batch_size, total_seq_len, total_seq_len]
        """
        batch_size = point_clouds.size(0)
        num_pc_tokens = self.num_pc_tokens
        text_seq_len = target_codes.size(1)
        total_seq_len = num_pc_tokens + text_seq_len
        
        # Create attention mask
        # Point cloud tokens can attend to all positions
        # Text tokens follow causal masking but can attend to all point cloud tokens
        attention_mask = torch.zeros(
            batch_size, total_seq_len, total_seq_len,
            dtype=torch.bool, device=point_clouds.device
        )
        
        # Point cloud tokens attend to everything (bidirectional)
        attention_mask[:, :num_pc_tokens, :] = True
        
        # Text tokens attend to point cloud tokens and previous text tokens (causal)
        for i in range(text_seq_len):
            text_pos = num_pc_tokens + i
            # Attend to all point cloud tokens
            attention_mask[:, text_pos, :num_pc_tokens] = True
            # Attend to current and previous text tokens (causal)
            attention_mask[:, text_pos, num_pc_tokens:text_pos + 1] = True
        
        return attention_mask
    
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get text embeddings from input token IDs.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Text embeddings [batch_size, seq_len, hidden_dim]
        """
        try:
            # Get embedding layer from LLM
            if hasattr(self.llm, 'get_input_embeddings'):
                embed_layer = self.llm.get_input_embeddings()
            elif hasattr(self.llm, 'embed_tokens'):
                embed_layer = self.llm.embed_tokens
            elif hasattr(self.llm, 'embeddings'):
                embed_layer = self.llm.embeddings.word_embeddings
            else:
                # Fallback: try to find embedding layer
                for module in self.llm.modules():
                    if isinstance(module, nn.Embedding) and module.num_embeddings == self.vocab_size:
                        embed_layer = module
                        break
                else:
                    raise AttributeError("Could not find embedding layer in LLM")
            
            return embed_layer(input_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to get text embeddings: {e}")
            raise RuntimeError(f"Text embedding failed: {e}")
    
    def forward(self, 
                point_clouds: torch.Tensor, 
                target_codes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training with negative log-likelihood loss.
        
        Args:
            point_clouds: Point cloud tensor [batch_size, num_points, 3]
            target_codes: Target code token IDs [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Negative log-likelihood loss
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If forward pass fails
        """
        # Validate inputs
        self._validate_inputs(point_clouds, target_codes)
        
        batch_size, seq_len = target_codes.shape
        
        try:
            # Use autocast for mixed precision if enabled
            with torch.cuda.amp.autocast(enabled=self.mixed_precision and point_clouds.is_cuda):
                # 1. Generate query tokens from point clouds
                query_tokens = self.projector(point_clouds)
                # Shape: [batch_size, num_pc_tokens, hidden_dim]
                
                # 2. Get text embeddings from target codes
                # For training, we use teacher forcing, so input is target_codes shifted
                input_ids = target_codes[:, :-1]  # Remove last token
                labels = target_codes[:, 1:]      # Remove first token
                
                text_embeddings = self._get_text_embeddings(input_ids)
                # Shape: [batch_size, seq_len-1, hidden_dim]
                
                # 3. Combine query tokens with text embeddings
                combined_embeddings = torch.cat([query_tokens, text_embeddings], dim=1)
                # Shape: [batch_size, num_pc_tokens + seq_len-1, hidden_dim]
                
                # 4. Create attention mask
                attention_mask = self._create_attention_mask(point_clouds, input_ids)
                # Shape: [batch_size, total_seq_len, total_seq_len]
                
                # 5. Forward pass through LLM
                # Use the LLM's transformer layers
                if hasattr(self.llm, 'transformer'):
                    transformer_outputs = self.llm.transformer(
                        inputs_embeds=combined_embeddings,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                elif hasattr(self.llm, 'model'):
                    transformer_outputs = self.llm.model(
                        inputs_embeds=combined_embeddings,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                else:
                    # Fallback: use the entire model
                    transformer_outputs = self.llm(
                        inputs_embeds=combined_embeddings,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                
                hidden_states = transformer_outputs.last_hidden_state
                # Shape: [batch_size, total_seq_len, hidden_dim]
                
                # 6. Get logits for text positions only (skip point cloud tokens)
                text_hidden_states = hidden_states[:, self.num_pc_tokens:, :]
                # Shape: [batch_size, seq_len-1, hidden_dim]
                
                logits = self.lm_head(text_hidden_states)
                # Shape: [batch_size, seq_len-1, vocab_size]
                
                # 7. Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(
                    logits.view(-1, self.vocab_size),
                    labels.view(-1)
                )
                
                return loss
                
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(f"Forward pass failed: {e}")
    
    def _sample_next_token(self, logits: torch.Tensor) -> int:
        """
        Sample next token from logits using configured sampling strategy.
        
        Args:
            logits: Token logits [vocab_size]
            
        Returns:
            int: Sampled token ID
        """
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Apply top-p (nucleus) sampling if enabled
        if self.do_sample and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            # Set logits to -inf for tokens to remove
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')