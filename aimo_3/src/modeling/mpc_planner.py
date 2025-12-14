"""
Model-Predictive Control (MPC) Planner using G-JEPA World Model.

Implements:
- Cross-Entropy Method (CEM) for trajectory optimization
- Uncertainty-aware planning
- Integration with MCTS search
- Parallel trajectory evaluation
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import numpy as np

from .gjepa_model import ActionConditionedGJEPA


class MPCPlanner:
    """
    Model-Predictive Control planner using learned world model.
    
    Uses Cross-Entropy Method (CEM) to optimize action sequences.
    """
    
    def __init__(
        self,
        world_model: ActionConditionedGJEPA,
        horizon: int = 10,
        num_candidates: int = 100,
        num_iterations: int = 5,
        elite_frac: float = 0.1,
        temperature: float = 1.0,
        uncertainty_penalty: float = 0.1,
    ):
        """
        Initialize MPC planner.
        
        Args:
            world_model: Trained world model for rollouts
            horizon: Planning horizon (number of steps)
            num_candidates: Number of candidate sequences per iteration
            num_iterations: Number of CEM iterations
            elite_frac: Fraction of top sequences to keep as elites
            temperature: Temperature for action sampling
            uncertainty_penalty: Penalty for high-uncertainty trajectories
        """
        self.world_model = world_model
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations
        self.elite_frac = elite_frac
        self.temperature = temperature
        self.uncertainty_penalty = uncertainty_penalty
        
        self.num_elite = max(1, int(num_candidates * elite_frac))
        self.num_actions = world_model.action_encoder.num_actions
    
    def plan(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        cost_fn: Optional[Callable] = None,
        avoid_uncertainty: bool = True,
    ) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Plan action sequence to reach goal.
        
        Args:
            current_state: Current state latent [D]
            goal_state: Goal state latent [D]
            cost_fn: Optional custom cost function(trajectory, goal) -> cost
                    Default: L2 distance to goal at final state
            avoid_uncertainty: Whether to penalize uncertain predictions
            
        Returns:
            (best_action_sequence, expected_cost, info_dict)
        """
        device = current_state.device
        
        # Initialize action distribution
        action_mean = torch.ones(self.horizon, device=device) * (self.num_actions / 2)
        action_std = torch.ones(self.horizon, device=device) * (self.num_actions / 4)
        
        best_sequence: torch.Tensor = action_mean.long()
        best_cost = float('inf')
        cost_history: List[float] = []
        
        for iteration in range(self.num_iterations):
            # Sample action sequences
            action_sequences = self._sample_actions(action_mean, action_std, device)
            
            # Evaluate all sequences
            costs, uncertainties = self._evaluate_sequences(
                current_state,
                goal_state,
                action_sequences,
                cost_fn,
                avoid_uncertainty,
            )
            
            # Select elite sequences
            elite_indices = torch.argsort(costs)[:self.num_elite]
            elite_sequences = action_sequences[elite_indices]
            elite_costs = costs[elite_indices]
            
            # Update action distribution (CEM update)
            action_mean = elite_sequences.float().mean(dim=0)
            action_std = elite_sequences.float().std(dim=0) + 1e-6
            
            # Track best
            if elite_costs[0] < best_cost:
                best_cost = elite_costs[0].item()
                best_sequence = elite_sequences[0]
            
            cost_history.append(elite_costs[0].item())
        
        info: Dict[str, Any] = {
            'cost_history': cost_history,
            'final_action_mean': action_mean.cpu().numpy(),
            'final_action_std': action_std.cpu().numpy(),
            'num_evaluations': self.num_candidates * self.num_iterations,
        }
        
        return best_sequence, best_cost, info
    
    def _sample_actions(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample action sequences from Gaussian distribution."""
        samples = torch.randn(
            self.num_candidates,
            self.horizon,
            device=device,
        ) * std.unsqueeze(0) * self.temperature + mean.unsqueeze(0)
        
        # Clamp to valid action range
        samples = torch.clamp(samples, 0, self.num_actions - 1).long()
        
        return samples
    
    def _evaluate_sequences(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        action_sequences: torch.Tensor,
        cost_fn: Optional[Callable],
        avoid_uncertainty: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate batch of action sequences.
        
        Returns:
            (costs, uncertainties)
        """
        costs_list: List[float] = []
        uncertainties_list: List[float] = []
        
        with torch.no_grad():
            for actions in action_sequences:
                # Rollout trajectory
                if self.world_model.use_distributional and avoid_uncertainty:
                    # Get uncertainty estimates
                    trajectory, uncertainty = self._rollout_with_uncertainty(
                        current_state,
                        actions,
                    )
                    total_uncertainty = uncertainty.mean().item()
                else:
                    trajectory = cast(torch.Tensor, self.world_model.rollout(
                        initial_state=current_state,
                        actions=actions.unsqueeze(0),
                        horizon=self.horizon,
                        stochastic=False,
                    ))
                    total_uncertainty = 0.0
                
                # Compute cost
                if cost_fn is not None:
                    cost = cost_fn(trajectory, goal_state)
                else:
                    # Default: L2 distance to goal at final state
                    final_state = trajectory[-1]
                    cost = torch.norm(final_state - goal_state, p=2).item()
                
                # Add uncertainty penalty
                if avoid_uncertainty:
                    cost += self.uncertainty_penalty * total_uncertainty
                
                costs_list.append(float(cost))
                uncertainties_list.append(float(total_uncertainty))
        
        return torch.tensor(costs_list, device=current_state.device), torch.tensor(uncertainties_list, device=current_state.device)
    
    def _rollout_with_uncertainty(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rollout trajectory and track uncertainty."""
        states = [initial_state]
        uncertainties = []
        
        for t in range(self.horizon):
            current_state = states[-1]
            current_action = actions[t].unsqueeze(0)
            
            h_seq = current_state.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            action_seq = current_action.unsqueeze(0)  # [1, 1]
            
            # Get distributional prediction
            mean, variance, _ = self.world_model.forward(
                h_seq,
                action_seq,
                return_distribution=True,
            )
            
            next_state = mean.squeeze()
            uncertainty = variance.mean().sqrt()
            
            states.append(next_state)
            uncertainties.append(uncertainty)
        
        trajectory = torch.stack(states)
        uncertainty_trajectory = torch.stack(uncertainties)
        
        return trajectory, uncertainty_trajectory
    
    def plan_with_constraints(
        self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        constraint_fn: Callable[[torch.Tensor], bool],
        max_retries: int = 10,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Plan with state constraints.
        
        Args:
            current_state: Current state
            goal_state: Goal state
            constraint_fn: Function that returns True if state satisfies constraints
            max_retries: Maximum planning attempts
            
        Returns:
            (action_sequence, cost) or (None, inf) if failed
        """
        for attempt in range(max_retries):
            sequence, cost, info = self.plan(current_state, goal_state)
            
            # Check if trajectory satisfies constraints
            trajectory = cast(torch.Tensor, self.world_model.rollout(
                initial_state=current_state,
                actions=sequence.unsqueeze(0),
                horizon=self.horizon,
            ))
            
            # Check all states in trajectory
            valid = True
            for state in trajectory:
                if not constraint_fn(state):
                    valid = False
                    break
            
            if valid:
                return sequence, cost
        
        # Failed to find valid plan
        return None, float('inf')


class MCTSIntegration:
    """
    Integration of MPC planner with Monte Carlo Tree Search.
    
    Uses world model rollouts as MCTS expansion heuristic.
    """
    
    def __init__(
        self,
        planner: MPCPlanner,
        rollout_depth: int = 5,
    ):
        """
        Initialize MCTS integration.
        
        Args:
            planner: MPC planner
            rollout_depth: Depth of world model rollouts for expansion
        """
        self.planner = planner
        self.rollout_depth = rollout_depth
    
    def expansion_heuristic(
        self,
        state: torch.Tensor,
        available_actions: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Score available actions using world model rollouts.
        
        Args:
            state: Current state
            available_actions: List of action indices
            
        Returns:
            List of (action, score) pairs, sorted by score
        """
        action_scores = []
        
        with torch.no_grad():
            for action in available_actions:
                # Short rollout with this action
                action_tensor = torch.tensor([action] * self.rollout_depth)
                trajectory = cast(torch.Tensor, self.planner.world_model.rollout(
                    initial_state=state,
                    actions=action_tensor.unsqueeze(0),
                    horizon=self.rollout_depth,
                ))
                
                # Score based on trajectory diversity (exploration value)
                trajectory_variance = trajectory.var(dim=0).mean().item()
                score = trajectory_variance
                
                action_scores.append((action, score))
        
        # Sort by score (descending)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        return action_scores


def create_mpc_planner(
    world_model: ActionConditionedGJEPA,
    **kwargs,
) -> MPCPlanner:
    """Factory function to create MPC planner."""
    return MPCPlanner(world_model=world_model, **kwargs)
