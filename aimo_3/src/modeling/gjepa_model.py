"""
G-JEPA World Model: Action-conditioned transformer-based world model for geometric reasoning.

Implements:
- Action-conditioned dynamics: s_{t+1} = f(s_t, a_t)
- Multi-step autoregressive rollout
- Distributional predictions with uncertainty quantification
- Model-predictive control planning
- Causal reasoning capabilities

Upgraded from basic masked prediction to full world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload
import math

from ..geometry.scene_encoder import SceneEncoder
from .action_encoder import ActionEncoder, TheoremType, create_action_encoder
from .distributional_head import DistributionalHead, EnsembleHead, create_distributional_head


class LiquidDynamics(nn.Module):
    """Simple liquid-style recurrent dynamics over latent sequences.

    This implements a continuous-time-inspired update:

        h_t = h_{t-1} + dt * (-(h_{t-1} - z_t) / tau)

    where z_t = tanh(W([x_t, h_{t-1}])), tau is a learnable time constant,
    and x_t is the per-step input (state + optional action embedding).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.Linear(hidden_dim * 2, hidden_dim)
        # Per-dimension time constants, initialized to 1.0
        self.tau = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Run liquid dynamics over a sequence.

        Args:
            x: Input sequence [B, T, H]
            dt: Time step (default 1.0)

        Returns:
            Tensor of shape [B, T, H] with hidden states per step.
        """
        B, T, H = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device, dtype=x.dtype)
        outputs: List[torch.Tensor] = []

        for t in range(T):
            inp = torch.cat([x[:, t, :], h], dim=-1)
            z = torch.tanh(self.cell(inp))
            # Continuous-time style update, discretized with dt
            h = h + dt * (-(h - z) / self.tau)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class ActionConditionedGJEPA(nn.Module):
    """
    Action-Conditioned Geometric Joint-Embedding Predictive Architecture.
    
    Full world model with:
    - Action conditioning for theorem applications
    - Multi-step trajectory rollout
    - Uncertainty quantification
    - Planning interface
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 100,
        num_actions: int = 51,
        use_distributional: bool = True,
        use_ensemble: bool = False,
        num_ensemble: int = 5,
        dropout: float = 0.1,
        legacy_mode: bool = False,
        use_liquid: bool = False,
    ):
        """
        Initialize action-conditioned G-JEPA world model.
        
        Args:
            latent_dim: Dimension of state latent vectors [D]
            action_dim: Dimension of action embeddings
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            num_actions: Number of discrete actions (theorem types)
            use_distributional: Use distributional predictions
            use_ensemble: Use ensemble for epistemic uncertainty
            num_ensemble: Number of ensemble members
            dropout: Dropout rate
            legacy_mode: Backward compatibility mode (no actions)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_distributional = use_distributional
        self.legacy_mode = legacy_mode
        self.use_liquid_dynamics = use_liquid
        
        # Action encoder
        if not legacy_mode:
            self.action_encoder = create_action_encoder(
                action_dim=action_dim,
                num_actions=num_actions,
                dropout=dropout,
            )
        
        # Input projection: [D] → [hidden_dim]
        self.state_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.02
        )
        
        # Cross-attention for action conditioning
        if not legacy_mode:
            self.action_cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
                kdim=action_dim,
                vdim=action_dim,
            )
            self.action_norm = nn.LayerNorm(hidden_dim)
        
        # Dynamics module: either Transformer or LiquidDynamics
        if self.use_liquid_dynamics:
            self.dynamics = LiquidDynamics(hidden_dim)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.dynamics = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        
        # Prediction head
        if use_distributional:
            self.predictor = create_distributional_head(
                input_dim=hidden_dim,
                output_dim=latent_dim,
                ensemble=use_ensemble,
                num_heads=num_ensemble if use_ensemble else 1,
                dropout=dropout,
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )
    
    @overload
    def forward(
        self,
        h_seq: torch.Tensor,
        actions: Optional[torch.Tensor],
        mask_indices: List[torch.Tensor],
        lengths: Optional[List[int]] = ...,
        return_distribution: bool = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def forward(
        self,
        h_seq: torch.Tensor,
        actions: Optional[torch.Tensor] = ...,
        mask_indices: None = ...,
        lengths: Optional[List[int]] = ...,
        return_distribution: Literal[True] = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def forward(
        self,
        h_seq: torch.Tensor,
        actions: Optional[torch.Tensor] = ...,
        mask_indices: None = ...,
        lengths: Optional[List[int]] = ...,
        return_distribution: Literal[False] = ...,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        h_seq: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        mask_indices: Optional[List[torch.Tensor]] = None,
        lengths: Optional[List[int]] = None,
        return_distribution: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass: predict next states or masked states.
        
        Args:
            h_seq: State sequence [batch_size, T, D]
            actions: Action sequence [batch_size, T] or [batch_size, T-1]
                     If None and legacy_mode=False, uses NO_ACTION
            mask_indices: For legacy masked prediction
            lengths: Sequence lengths
            return_distribution: Return (mean, variance) instead of point estimate
            
        Returns:
            For next-step prediction (actions provided):
                (predicted_states, target_states) or (mean, variance, targets)
            For masked prediction (mask_indices provided):
                (predicted_latents, target_latents)
        """
        batch_size, seq_len, D = h_seq.shape
        
        # Project states to hidden dimension
        x = self.state_proj(h_seq)  # [B, T, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Action conditioning
        if not self.legacy_mode and actions is not None:
            # Encode actions
            action_emb = self.action_encoder(actions)  # [B, T] → [B, T, action_dim]
            
            # Cross-attention: attend to actions
            x_attn, _ = self.action_cross_attn(
                query=x,
                key=action_emb,
                value=action_emb,
            )
            x = self.action_norm(x + x_attn)
        
        # Encode with chosen dynamics module
        encoded = self.dynamics(x)  # [B, T, hidden_dim]
        
        # Prediction mode: next-step vs masked
        if mask_indices is not None:
            # Legacy masked prediction
            return self._masked_prediction(encoded, h_seq, mask_indices, lengths, return_distribution)
        else:
            # Next-step prediction
            return self._next_step_prediction(encoded, h_seq, return_distribution)
    
    def _next_step_prediction(
        self,
        encoded: torch.Tensor,
        h_seq: torch.Tensor,
        return_distribution: bool,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Predict next states from action-conditioned encoding."""
        # Predict h_{t+1} from h_t + a_t encoding
        # Use all but last position to predict next
        if encoded.size(1) > 1:
            context = encoded[:, :-1]  # [B, T-1, hidden_dim]
            targets = h_seq[:, 1:]      # [B, T-1, D]
        else:
            context = encoded
            targets = h_seq
        
        if self.use_distributional:
            if isinstance(self.predictor, EnsembleHead):
                mean, aleatoric_var, epistemic_var = self.predictor(context)
                total_var = aleatoric_var + epistemic_var
                if return_distribution:
                    return mean, total_var, targets
                else:
                    return mean, targets
            else:
                mean, variance, _ = self.predictor(context)
                if return_distribution:
                    return mean, variance, targets
                else:
                    return mean, targets
        else:
            predictions = self.predictor(context)
            return predictions, targets
    
    def _masked_prediction(
        self,
        encoded: torch.Tensor,
        h_seq: torch.Tensor,
        mask_indices: List[torch.Tensor],
        lengths: Optional[List[int]],
        return_distribution: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy masked prediction (for backward compatibility)."""
        batch_size, seq_len, D = h_seq.shape
        predicted_latents: list[torch.Tensor] = []
        target_latents: list[torch.Tensor] = []
        
        for b, mask_idx in enumerate(mask_indices):
            if len(mask_idx) > 0:
                if isinstance(mask_idx, torch.Tensor):
                    mask_idx_tensor = mask_idx
                else:
                    mask_idx_tensor = torch.tensor(mask_idx, dtype=torch.long, device=h_seq.device)
                
                valid_len = lengths[b] if lengths is not None else seq_len
                
                # Create masks
                mask_bool = torch.zeros(seq_len, dtype=torch.bool, device=h_seq.device)
                mask_bool[mask_idx_tensor] = True
                
                padding_bool = torch.zeros(seq_len, dtype=torch.bool, device=h_seq.device)
                if valid_len < seq_len:
                    padding_bool[valid_len:] = True
                
                context_bool = ~mask_bool & ~padding_bool
                unmasked_idx = context_bool.nonzero(as_tuple=False).squeeze(-1)
                
                if len(unmasked_idx) > 0:
                    context = encoded[b, unmasked_idx].mean(dim=0)
                else:
                    valid_idx = (~padding_bool).nonzero(as_tuple=False).squeeze(-1)
                    if len(valid_idx) > 0:
                        context = encoded[b, valid_idx].mean(dim=0)
                    else:
                        context = encoded[b].mean(dim=0)
                
                # Predict for each masked position
                for idx in mask_idx_tensor:
                    idx_int = int(idx.item())
                    
                    if self.use_distributional and return_distribution:
                        mean, variance, _ = self.predictor(context.unsqueeze(0))
                        predicted_latents.append(mean.squeeze(0))
                    else:
                        predicted = self.predictor(context.unsqueeze(0)).squeeze(0)
                        predicted_latents.append(predicted)
                    
                    target = h_seq[b, idx_int]
                    target_latents.append(target)
        
        if not predicted_latents:
            dummy = torch.zeros(1, self.latent_dim, device=h_seq.device)
            return dummy, dummy
        
        predicted = torch.stack(predicted_latents)
        targets = torch.stack(target_latents)
        return predicted, targets
    
    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        stochastic: bool = False,
        num_samples: int = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Autoregressive multi-step rollout.
        
        Args:
            initial_state: Initial state latent [D] or [B, D]
            actions: Action sequence [horizon] or [B, horizon]
            horizon: Number of steps to rollout
            stochastic: Whether to sample from distribution
            num_samples: Number of samples if stochastic
            
        Returns:
            State trajectory [horizon+1, D] or [B, horizon+1, D]
            If stochastic: (mean_trajectory, variance_trajectory)
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)  # [1, D]
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)  # [1, horizon]
        
        batch_size = initial_state.size(0)
        assert actions.size(0) == batch_size
        assert actions.size(1) >= horizon
        
        states = [initial_state]
        
        for t in range(horizon):
            # Current state and action
            current_state = states[-1]  # [B, D]
            current_action = actions[:, t]  # [B]
            
            # Prepare input sequence (add time dimension)
            h_seq = current_state.unsqueeze(1)  # [B, 1, D]
            action_seq = current_action.unsqueeze(1)  # [B, 1]
            
            # Predict next state
            with torch.set_grad_enabled(self.training):
                if self.use_distributional:
                    mean, variance, _ = self.forward(
                        h_seq, action_seq, return_distribution=True
                    )
                    mean = mean.squeeze(1)  # [B, D]
                    variance = variance.squeeze(1)
                    
                    if stochastic:
                        # Sample from distribution
                        std = torch.sqrt(variance)
                        epsilon = torch.randn_like(mean)
                        next_state = mean + std * epsilon
                    else:
                        next_state = mean
                else:
                    next_state, _ = self.forward(h_seq, action_seq, return_distribution=False)
                    next_state = next_state.squeeze(1)
            
            states.append(next_state)
        
        trajectory = torch.stack(states, dim=1)  # [B, horizon+1, D]
        
        if batch_size == 1:
            trajectory = trajectory.squeeze(0)  # [horizon+1, D]
        
        return trajectory
    
    def plan(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        horizon: int = 10,
        num_candidates: int = 100,
        num_iterations: int = 5,
        elite_frac: float = 0.1,
    ) -> Tuple[torch.Tensor, float]:
        """
        Model-Predictive Control planning using Cross-Entropy Method.
        
        Args:
            current_state: Current state latent [D]
            goal_state: Goal state latent [D]
            horizon: Planning horizon
            num_candidates: Number of action sequences to sample
            num_iterations: CEM iterations
            elite_frac: Fraction of top sequences to keep
            
        Returns:
            (best_action_sequence, expected_cost)
        """
        device = current_state.device
        num_elite = max(1, int(num_candidates * elite_frac))
        
        # Initialize action distribution (uniform over action space)
        action_mean = torch.ones(horizon, device=device) * (self.action_encoder.num_actions // 2)
        action_std = torch.ones(horizon, device=device) * (self.action_encoder.num_actions // 4)
        
        best_sequence: torch.Tensor = action_mean.long()
        best_cost = float('inf')
        
        for iteration in range(num_iterations):
            # Sample action sequences
            action_sequences_list: list[torch.Tensor] = []
            for _ in range(num_candidates):
                actions = torch.clamp(
                    torch.randn(horizon, device=device) * action_std + action_mean,
                    0,
                    self.action_encoder.num_actions - 1
                ).long()
                action_sequences_list.append(actions)
            
            action_sequences = torch.stack(action_sequences_list)  # [num_candidates, horizon]
            
            # Evaluate sequences
            costs_list: list[float] = []
            for actions in action_sequences:
                trajectory = self.rollout(
                    initial_state=current_state,
                    actions=actions.unsqueeze(0),
                    horizon=horizon,
                    stochastic=False,
                )
                
                # Cost: distance to goal at final state
                final_state = trajectory[-1]
                cost = torch.norm(final_state - goal_state, p=2).item()
                costs_list.append(float(cost))
            
            costs_t = torch.tensor(costs_list, device=device)
            
            # Select elite sequences
            elite_indices = torch.argsort(costs_t)[:num_elite]
            elite_sequences = action_sequences[elite_indices]
            elite_costs = costs_t[elite_indices]
            
            # Update distribution
            action_mean = elite_sequences.float().mean(dim=0)
            action_std = elite_sequences.float().std(dim=0) + 1e-6
            
            # Track best
            if elite_costs[0] < best_cost:
                best_cost = elite_costs[0].item()
                best_sequence = elite_sequences[0]
        
        return best_sequence, best_cost
    
    def compute_loss(
        self,
        h_seq: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        mask_indices: Optional[List[torch.Tensor]] = None,
        lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute loss for training.
        
        Returns:
            Loss scalar or dict of losses
        """
        if self.use_distributional:
            if mask_indices is not None:
                # Legacy masked prediction
                predicted, targets = self.forward(h_seq, actions, mask_indices, lengths)
                loss = F.mse_loss(predicted, targets)
            else:
                # Next-step prediction with NLL
                mean, variance, targets = self.forward(
                    h_seq, actions, return_distribution=True
                )
                
                # Reshape for loss
                mean_flat = mean.view(-1, self.latent_dim)
                var_flat = variance.view(-1, self.latent_dim)
                targets_flat = targets.view(-1, self.latent_dim)
                
                # Gaussian NLL (variance predicted by distributional head / ensemble)
                var_flat = torch.clamp(var_flat, min=1e-8)
                squared_error = (targets_flat - mean_flat) ** 2
                nll = 0.5 * (math.log(2 * math.pi) + torch.log(var_flat) + squared_error / var_flat)
                loss = nll.mean()
            
            return loss
        else:
            predicted, targets = self.forward(h_seq, actions, mask_indices, lengths, return_distribution=False)
            loss = F.mse_loss(predicted, targets)
            return loss
    
    def compute_heuristic_score(
        self,
        current_state: Any,
        candidate_state: Any,
        goal_state: Optional[Any] = None,
    ) -> float:
        """Compute heuristic score for search (backward compatible)."""
        encoder = getattr(self, "encoder", None)
        if encoder is None:
            return 0.0
        
        self.eval()
        encoder.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            h_current = encoder.encode_state(current_state).to(device)
            h_candidate = encoder.encode_state(candidate_state).to(device)
            h_seq = torch.stack([h_current, h_candidate], dim=0).unsqueeze(0)
            
            # Use NO_ACTION as default action
            if not self.legacy_mode:
                actions = torch.tensor([[TheoremType.NO_ACTION.value]], device=device)
                loss = self.compute_loss(h_seq, actions).item()
            else:
                mask_indices = [torch.tensor([1], dtype=torch.long, device=device)]
                lengths = [2]
                loss = self.compute_loss(h_seq, None, mask_indices, lengths).item()
            
            score = -loss
            
            if goal_state is not None:
                h_goal = encoder.encode_state(goal_state).to(device)
                dist = torch.norm(h_candidate - h_goal, p=2).item()
                score = score - 0.1 * dist
        
        return float(score)


# Legacy GJEPA for backward compatibility
class GJEPA(ActionConditionedGJEPA):
    """Legacy G-JEPA interface (masked prediction only)."""
    
    def __init__(self, **kwargs):
        kwargs['legacy_mode'] = True
        kwargs['use_distributional'] = False
        super().__init__(**kwargs)


def create_gjepa_model(
    latent_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    max_seq_len: int = 100,
    action_conditioned: bool = False,
    **kwargs,
) -> Union[ActionConditionedGJEPA, GJEPA]:
    """
    Factory function to create G-JEPA model.
    
    Args:
        latent_dim: Latent dimension [D]
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        action_conditioned: Use action-conditioned world model
        **kwargs: Additional arguments
        
    Returns:
        ActionConditionedGJEPA or legacy GJEPA
    """
    if action_conditioned:
        return ActionConditionedGJEPA(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    else:
        return GJEPA(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )
