"""
Causal Reasoning Module for G-JEPA World Model.

Implements:
- Interventional predictions: p(s_{t+1} | do(a_t), s_t)
- Counterfactual reasoning: "What if we applied action a' instead?"
- Causal graph learning from proof traces
- Confound detection and adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx

from .gjepa_model import ActionConditionedGJEPA


class CausalReasoner(nn.Module):
    """
    Causal reasoning module for interventional and counterfactual inference.
    
    Uses structural causal model (SCM) framework with learned dynamics.
    """
    
    def __init__(
        self,
        world_model: ActionConditionedGJEPA,
        latent_dim: int = 256,
        action_dim: int = 256,
        freeze_world_model: bool = True,
    ):
        """
        Initialize causal reasoner.
        
        Args:
            world_model: Trained G-JEPA world model
            latent_dim: Dimension of state latents
            action_dim: Dimension of action embeddings
            freeze_world_model: Whether to freeze world model parameters
        """
        super().__init__()
        self.world_model = world_model
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        if freeze_world_model:
            for param in self.world_model.parameters():
                param.requires_grad = False
        
        # Intervention head: models direct causal effect of actions
        # s_{t+1} = f(s_t) + g(a_t) + ε
        self.intervention_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # Counterfactual memory: stores (s, a, s') tuples for learning
        self.counterfactual_memory: List[Dict[str, torch.Tensor]] = []
        self.max_memory_size = 10000
        
       # Learned causal graph (adjacency matrix over latent dimensions)
        self.causal_adjacency = nn.Parameter(
            torch.eye(latent_dim) * 0.1,
            requires_grad=True,
        )
    
    def intervene(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        confounders: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interventional prediction: p(s_{t+1} | do(a_t), s_t).
        
        do(a_t) represents setting action to a_t, cutting all incoming edges
        to a_t in the causal graph.
        
        Args:
            state: Current state latent [D] or [B, D]
            action: Intervention action [1] or [B]
            confounders: Optional confounder variables
            
        Returns:
            Predicted next state under intervention
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        # Encode action
        action_emb = self.world_model.action_encoder(action)  # [B, action_dim]
        
        # Compute interventional effect
        combined = torch.cat([state, action_emb], dim=-1)
        intervention_effect = self.intervention_head(combined)
        
        # Apply causal graph constraints
        # Only allow causal parents to affect outcome
        causal_mask = (self.causal_adjacency > 0.1).float()
        state_contribution = state @ causal_mask
        
        # Combine natural dynamics + intervention
        # Natural: s_{t+1} = f(s_t)
        # Intervention: s_{t+1} = f(s_t) + g(do(a_t))
        next_state = state_contribution + intervention_effect
        
        return next_state
    
    def counterfactual(
        self,
        actual_state: torch.Tensor,
        actual_action: torch.Tensor,
        actual_next_state: torch.Tensor,
        counterfactual_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Counterfactual reasoning: "What if we had applied action a' instead?"
        
        Computes three steps:
        1. Abduction: Infer exogenous noise U given (s_t, a_t, s_{t+1})
        2. Action: Replace a_t with a'_t
        3. Prediction: Compute s'_{t+1} using modified action and inferred U
        
        Args:
            actual_state: Actual state s_t
            actual_action: Actual action a_t
            actual_next_state: Actual next state s_{t+1}
            counterfactual_action: Counterfactual action a'_t
            
        Returns:
            (counterfactual_next_state, residual_noise)
        """
        # Step 1: Abduction - infer noise
        # U = s_{t+1} - f(s_t, a_t)
        predicted_actual = self.intervene(actual_state, actual_action)
        residual = actual_next_state - predicted_actual  # Inferred noise U
        
        # Step 2 & 3: Action + Prediction
        # s'_{t+1} = f(s_t, a'_t) + U
        predicted_counterfactual = self.intervene(actual_state, counterfactual_action)
        counterfactual_next_state = predicted_counterfactual + residual
        
        return counterfactual_next_state, residual
    
    def add_to_memory(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ):
        """Add experience tuple to counterfactual memory."""
        experience = {
            'state': state.detach().cpu(),
            'action': action.detach().cpu(),
            'next_state': next_state.detach().cpu(),
        }
        
        self.counterfactual_memory.append(experience)
        
        # Keep memory size bounded
        if len(self.counterfactual_memory) > self.max_memory_size:
            self.counterfactual_memory.pop(0)
    
    def learn_causal_graph(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        sparsity_penalty: float = 0.01,
    ) -> torch.Tensor:
        """
        Learn causal graph structure from data.
        
        Uses continuous optimization with sparsity regularization.
        
        Args:
            states: State sequence [T, D]
            actions: Action sequence [T]
            next_states: Next state sequence [T, D]
            sparsity_penalty: L1 penalty on causal adjacency
            
        Returns:
            Loss for causal graph learning
        """
        # Predict using current causal graph
        predicted_list: List[torch.Tensor] = []
        for i in range(len(states)):
            pred = self.intervene(states[i], actions[i])
            predicted_list.append(pred)
        
        predicted = torch.stack(predicted_list)
        
        # Prediction loss
        prediction_loss = F.mse_loss(predicted, next_states)
        
        # Sparsity penalty (encourage sparse causal graphs)
        sparsity_loss = torch.abs(self.causal_adjacency).sum() * sparsity_penalty
        
        # DAG constraint (encourage acyclicity)
        # tr(e^A) - d should be close to 0 for DAG
        expm = torch.matrix_exp(self.causal_adjacency * self.causal_adjacency.t())
        dag_loss = torch.trace(expm) - self.latent_dim
        
        total_loss = prediction_loss + sparsity_loss + 0.1 * (dag_loss ** 2)
        
        return total_loss
    
    def explain_prediction(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain which latent dimensions are most causal for the prediction.
        
        Args:
            state: State to explain
            action: Action to explain
            top_k: Number of top causal factors to return
            
        Returns:
            Dictionary with explanation
        """
        # Compute interventional prediction
        next_state = self.intervene(state, action)
        
        # Compute attribution: which state dimensions matter most?
        state_for_grad = state.detach().clone().requires_grad_(True)
        next_state_for_grad = self.intervene(state_for_grad, action)
        next_state_sum = next_state_for_grad.sum()
        next_state_sum.backward()
        
        grad = state_for_grad.grad
        assert grad is not None
        attributions = grad.abs()
        
        # Get top-k most important dimensions
        top_indices = torch.topk(attributions, k=min(top_k, len(attributions))).indices
        
        # Get causal parents for each dimension
        causal_structure = (self.causal_adjacency > 0.1).float()
        
        explanation = {
            'top_causal_dimensions': top_indices.tolist(),
            'attribution_scores': attributions[top_indices].tolist(),
            'causal_graph_density': (causal_structure.sum() / (self.latent_dim ** 2)).item(),
            'predicted_next_state': next_state.detach(),
        }
        
        return explanation


def create_causal_reasoner(
    world_model: ActionConditionedGJEPA,
    **kwargs,
) -> CausalReasoner:
    """Factory function to create causal reasoner."""
    return CausalReasoner(world_model=world_model, **kwargs)
