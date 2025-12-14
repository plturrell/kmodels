"""
Standalone test runner using unittest.
Adapts pytest-style tests to unittest or runs equivalent checks.
"""

import unittest
import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modeling.action_encoder import TheoremType, create_action_encoder
from src.modeling.distributional_head import DistributionalHead, EnsembleHead
from src.modeling.gjepa_model import create_gjepa_model
from src.modeling.causal_reasoner import CausalReasoner
from src.modeling.mpc_planner import MPCPlanner


class TestWorldModel(unittest.TestCase):
    
    def test_action_encoder(self):
        print("\nTesting Action Encoder...")
        encoder = create_action_encoder(action_dim=256)
        actions = torch.tensor([0, 5, 10])
        embeddings = encoder(actions)
        self.assertEqual(embeddings.shape, (3, 256))
        print("✓ Discrete action encoding")
        
        # Test sequence
        seq = [TheoremType.PYTHAGOREAN, TheoremType.TRIANGLE_SUM]
        emb = encoder.encode_sequence(seq)
        self.assertEqual(emb.shape, (2, 256))
        print("✓ Sequence encoding")

    def test_distributional_head(self):
        print("\nTesting Distributional Head...")
        head = DistributionalHead(input_dim=512, output_dim=256, num_components=1)
        x = torch.randn(10, 512)
        mean, variance, _ = head(x)
        self.assertEqual(mean.shape, (10, 256))
        self.assertTrue((variance > 0).all())
        print("✓ Gaussian prediction")

    def test_gjepa_model(self):
        print("\nTesting G-JEPA World Model...")
        model = create_gjepa_model(
            latent_dim=256,
            action_dim=256,
            hidden_dim=512,
            action_conditioned=True,
            use_distributional=True
        )
        
        # Forward pass
        h_seq = torch.randn(4, 10, 256)
        actions = torch.randint(0, 51, (4, 9))
        mean, targets = model(h_seq, actions)
        self.assertEqual(mean.shape, (4, 9, 256))
        print("✓ Forward pass")
        
        # Rollout
        initial = torch.randn(256)
        rollout_actions = torch.randint(0, 51, (10,))
        traj = model.rollout(initial, rollout_actions, horizon=10)
        self.assertEqual(traj.shape, (11, 256))
        print("✓ Multi-step rollout")
        
        # Planning
        print("\nTesting Planner...")
        current = torch.randn(256)
        goal = torch.randn(256)
        seq, cost = model.plan(current, goal, horizon=5, num_candidates=10, num_iterations=2)
        self.assertEqual(seq.shape, (5,))
        print(f"✓ Planning (cost: {cost:.4f})")

    def test_mpc_planner(self):
        print("\nTesting Advanced MPC Planner...")
        model = create_gjepa_model(
            latent_dim=256,
            action_conditioned=True,
        )
        planner = MPCPlanner(model, horizon=5, num_candidates=10, num_iterations=2)
        
        current = torch.randn(256)
        goal = torch.randn(256)
        
        seq, cost, info = planner.plan(current, goal)
        self.assertEqual(seq.shape, (5,))
        print("✓ MPC Planning execution")

    def test_causal_reasoner(self):
        print("\nTesting Causal Reasoner...")
        world_model = create_gjepa_model(latent_dim=256, action_conditioned=True)
        reasoner = CausalReasoner(world_model)
        
        state = torch.randn(256)
        action = torch.tensor([TheoremType.PYTHAGOREAN.value])
        next_state = reasoner.intervene(state, action)
        self.assertEqual(next_state.shape, (1, 256))
        print("✓ Interventional prediction")


if __name__ == '__main__':
    unittest.main()
