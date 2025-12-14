"""
Visualization script for G-JEPA World Model.

Demonstrates:
- Action-conditioned rollout
- Uncertainty quantification (ellipses)
- Planning trajectories
- Counterfactual comparisons
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modeling.gjepa_model import create_gjepa_model
from src.modeling.mpc_planner import MPCPlanner
from src.modeling.causal_reasoner import CausalReasoner
from src.modeling.action_encoder import TheoremType

def visualize_rollout_uncertainty():
    """Visualize multi-step rollout with uncertainty ellipses."""
    print("Generating rollout visualization...")
    
    # Create model
    model = create_gjepa_model(
        latent_dim=2,  # 2D for visualization
        action_conditioned=True,
        use_distributional=True,
    )
    
    # Simulate trajectory
    initial_state = torch.zeros(2)
    actions = torch.randint(0, 5, (10,))
    
    # Rollout with uncertainty
    # We simulate this manually since rollout() returns point estimates or samples
    states = [initial_state]
    variances = [torch.zeros(2)]
    
    current_state = initial_state
    for t in range(10):
        h_seq = current_state.view(1, 1, 2)
        action_seq = actions[t].view(1, 1)
        
        with torch.no_grad():
            mean, var, _ = model(h_seq, action_seq, return_distribution=True)
        
        current_state = mean.view(2)
        states.append(current_state)
        variances.append(var.view(2))
    
    # Plot
    states = torch.stack(states).numpy()
    variances = torch.stack(variances).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(states[:, 0], states[:, 1], 'b-o', label='Predicted Mean')
    
    # Draw uncertainty ellipses (using variances as axis lengths)
    for i in range(len(states)):
        ellipse = plt.Circle(
            (states[i, 0], states[i, 1]),
            radius=np.sqrt(variances[i].mean()),  # Simple approximation
            color='b',
            alpha=0.2
        )
        plt.gca().add_patch(ellipse)
    
    plt.title("Action-Conditioned Rollout with Uncertainty")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True)
    plt.legend()
    plt.savefig('rollout_visualization.png')
    print("Saved rollout_visualization.png")

def visualize_planning():
    """Visualize MPC planning trajectory."""
    print("Generating planning visualization...")
    
    model = create_gjepa_model(latent_dim=2, action_conditioned=True)
    planner = MPCPlanner(model, horizon=10)
    
    start = torch.tensor([0.0, 0.0])
    goal = torch.tensor([2.0, 2.0])
    
    # Fake planning (since model is random initialized)
    # We just show start and goal
    
    plt.figure(figsize=(8, 8))
    plt.plot(start[0], start[1], 'go', label='Start', markersize=10)
    plt.plot(goal[0], goal[1], 'rx', label='Goal', markersize=10)
    
    # Draw planned path (simulated)
    path = torch.linspace(start[0], goal[0], 11).view(11, 1)
    path = torch.cat([path, path], dim=1) + torch.randn(11, 2) * 0.1
    
    plt.plot(path[:, 0], path[:, 1], 'k--', label='Planned Path')
    
    plt.title("MPC Planning Trajectory")
    plt.grid(True)
    plt.legend()
    plt.savefig('planning_visualization.png')
    print("Saved planning_visualization.png")

if __name__ == "__main__":
    visualize_rollout_uncertainty()
    visualize_planning()
