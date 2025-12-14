"""
Action Encoder for G-JEPA World Model.

Encodes theorem applications and geometric operations as action embeddings
for action-conditioned world model prediction.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Union
from enum import Enum


class TheoremType(Enum):
    """Enumeration of geometric theorems and operations."""
    # Triangle theorems
    PYTHAGOREAN = 0
    TRIANGLE_SUM = 1
    SIMILAR_TRIANGLES = 2
    CONGRUENT_TRIANGLES = 3
    TRIANGLE_INEQUALITY = 4
    
    # Circle theorems
    INSCRIBED_ANGLE = 5
    CENTRAL_ANGLE = 6
    TANGENT_PERPENDICULAR = 7
    CHORD_PROPERTIES = 8
    CIRCLE_AREA = 9
    
    # Angle theorems
    VERTICAL_ANGLES = 10
    ALTERNATE_ANGLES = 11
    CORRESPONDING_ANGLES = 12
    SUPPLEMENTARY_ANGLES = 13
    COMPLEMENTARY_ANGLES = 14
    
    # Distance and coordinate theorems
    DISTANCE_FORMULA = 15
    MIDPOINT_FORMULA = 16
    SLOPE_FORMULA = 17
    SECTION_FORMULA = 18
    
    # Area and perimeter
    TRIANGLE_AREA = 19
    RECTANGLE_AREA = 20
    CIRCLE_CIRCUMFERENCE = 21
    POLYGON_AREA = 22
    
    # Advanced theorems
    POWER_OF_POINT = 23
    STEWARTS_THEOREM = 24
    CEVA_THEOREM = 25
    MENELAUS_THEOREM = 26
    PTOLEMY_THEOREM = 27
    
    # Geometric constructions
    ANGLE_BISECTOR = 28
    PERPENDICULAR_BISECTOR = 29
    PARALLEL_CONSTRUCTION = 30
    CIRCLE_CONSTRUCTION = 31
    
    # Trigonometry
    SINE_RULE = 32
    COSINE_RULE = 33
    TANGENT_RULE = 34
    
    # Coordinate geometry
    LINE_EQUATION = 35
    CIRCLE_EQUATION = 36
    DISTANCE_POINT_TO_LINE = 37
    
    # Transformations
    TRANSLATION = 38
    ROTATION = 39
    REFLECTION = 40
    SCALING = 41
    
    # Logic operations
    SUBSTITUTION = 42
    DEDUCTION = 43
    CONTRADICTION = 44
    INDUCTION = 45
    
    # Algebraic operations
    ALGEBRAIC_MANIPULATION = 46
    SIMPLIFICATION = 47
    FACTORIZATION = 48
    
    # Meta actions
    NO_ACTION = 49
    UNKNOWN_ACTION = 50


class ActionEncoder(nn.Module):
    """
    Encodes geometric theorem applications as embeddings.
    
    Supports:
    - Discrete action vocabulary (theorem types)
    - Continuous action parameters (theorem-specific arguments)
    - Action sequences
    - Batched encoding
    """
    
    def __init__(
        self,
        action_dim: int = 256,
        num_actions: int = 51,  # TheoremType enum size
        max_params: int = 8,
        use_param_encoding: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize action encoder.
        
        Args:
            action_dim: Dimension of action embeddings
            num_actions: Number of discrete actions in vocabulary
            max_params: Maximum number of parameters per action
            use_param_encoding: Whether to encode action parameters
            dropout: Dropout rate
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.max_params = max_params
        self.use_param_encoding = use_param_encoding
        
        # Discrete action embedding lookup table
        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_dim,
        )
        
        # Parameter encoder (for theorem-specific continuous parameters)
        if use_param_encoding:
            self.param_encoder = nn.Sequential(
                nn.Linear(max_params, action_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(action_dim, action_dim),
            )
            
            # Fusion layer to combine discrete + continuous
            self.fusion = nn.Sequential(
                nn.Linear(action_dim * 2, action_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(action_dim, action_dim),
            )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(action_dim)
    
    def forward(
        self,
        actions: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode actions to embeddings.
        
        Args:
            actions: Tensor of action indices, shape [batch_size] or [batch_size, seq_len]
            params: Optional action parameters, shape [batch_size, max_params] or
                   [batch_size, seq_len, max_params]
                   
        Returns:
            Action embeddings, shape [batch_size, action_dim] or 
            [batch_size, seq_len, action_dim]
        """
        # Embed discrete actions
        action_emb = self.action_embedding(actions)  # [B, D] or [B, T, D]
        
        # Add parameter encoding if provided
        if self.use_param_encoding and params is not None:
            param_emb = self.param_encoder(params)  # [B, D] or [B, T, D]
            
            # Concatenate and fuse
            combined = torch.cat([action_emb, param_emb], dim=-1)  # [B, 2D] or [B, T, 2D]
            action_emb = self.fusion(combined)  # [B, D] or [B, T, D]
        
        # Normalize
        action_emb = self.layer_norm(action_emb)
        
        return action_emb
    
    def encode_sequence(
        self,
        action_sequence: List[Union[int, TheoremType]],
        param_sequence: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode a sequence of actions.
        
        Args:
            action_sequence: List of action indices or TheoremType enums
            param_sequence: Optional list of parameter tensors
            device: Device to put tensors on
            
        Returns:
            Action sequence embeddings, shape [seq_len, action_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Convert TheoremType enums to indices
        action_ids: List[int] = []
        for action in action_sequence:
            if isinstance(action, TheoremType):
                action_ids.append(action.value)
            else:
                action_ids.append(int(action))
        
        actions_tensor = torch.tensor(action_ids, dtype=torch.long, device=device)
        
        # Handle parameters
        if param_sequence is not None:
            params = torch.stack([
                self._pad_params(p, device) for p in param_sequence
            ])
        else:
            params = None
        
        return self.forward(actions_tensor, params)
    
    def _pad_params(
        self,
        params: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Pad parameters to max_params length."""
        if len(params) < self.max_params:
            padding = torch.zeros(
                self.max_params - len(params),
                device=device,
            )
            params = torch.cat([params, padding])
        return params[:self.max_params]
    
    def get_action_name(self, action_idx: int) -> str:
        """Get human-readable name for action index."""
        try:
            return TheoremType(action_idx).name
        except ValueError:
            return f"UNKNOWN_ACTION_{action_idx}"


def create_action_encoder(
    action_dim: int = 256,
    **kwargs,
) -> ActionEncoder:
    """
    Factory function to create action encoder.
    
    Args:
        action_dim: Dimension of action embeddings
        **kwargs: Additional arguments passed to ActionEncoder
        
    Returns:
        Initialized ActionEncoder
    """
    return ActionEncoder(action_dim=action_dim, **kwargs)
