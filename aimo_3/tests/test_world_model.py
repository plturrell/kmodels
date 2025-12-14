"""
Comprehensive tests for action-conditioned G-JEPA world model.

Tests:
- Action encoding
- Distributional predictions
- Multi-step rollout
- Planning
- Uncertainty quantification
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest

from src.modeling.action_encoder import ActionEncoder, TheoremType, create_action_encoder
from src.modeling.distributional_head import DistributionalHead, EnsembleHead, create_distributional_head
from src.modeling.gjepa_model import ActionConditionedGJEPA, create_gjepa_model
from src.modeling.causal_reasoner import CausalReasoner
from src.modeling.mpc_planner import MPCPlanner
from src.geometry.scene_encoder import SceneEncoder


class TestActionEncoder:
    """Test action encoder functionality."""
    
    def test_discrete_action_encoding(self):
        """Test basic discrete action encoding."""
        encoder = create_action_encoder(action_dim=256)
        
        actions = torch.tensor([0, 5, 10])
        embeddings= encoder(actions)
        
        assert embeddings.shape == (3, 256)
        assert not torch.isnan(embeddings).any()
    
    def test_action_with_parameters(self):
        """Test action encoding with parameters."""
        encoder = create_action_encoder(action_dim=256, use_param_encoding=True)
        
        actions = torch.tensor([TheoremType.PYTHAGOREAN.value])
        params = torch.randn(1, 8)  # 8 parameters
        
        embeddings = encoder(actions, params)
        
        assert embeddings.shape == (1, 256)
    
    def test_sequence_encoding(self):
        """Test encoding action sequences."""
        encoder = create_action_encoder()
        
        action_sequence = [
            TheoremType.PYTHAGOREAN,
            TheoremType.TRIANGLE_SUM,
            TheoremType.SIMILAR_TRIANGLES,
        ]
        
        embeddings = encoder.encode_sequence(action_sequence)
        
        assert embeddings.shape == (3, 256)


class TestDistributionalHead:
    """Test distributional prediction head."""
    
    def test_gaussian_prediction(self):
        """Test single Gaussian prediction."""
        head = DistributionalHead(input_dim=512, output_dim=256, num_components=1)
        
        x = torch.randn(10, 512)
        mean, variance, mixture_weights = head(x)
        
        assert mean.shape == (10, 256)
        assert variance.shape == (10, 256)
        assert mixture_weights is None
        assert (variance > 0).all()  # Variance should be positive
    
    def test_mixture_prediction(self):
        """Test mixture of Gaussians prediction."""
        head = DistributionalHead(input_dim=512, output_dim=256, num_components=3)
        
        x = torch.randn(10, 512)
        means, variances, mixture_weights = head(x)
        
        assert means.shape == (10, 3, 256)
        assert variances.shape == (10, 3, 256)
        assert mixture_weights.shape == (10, 3)
        assert torch.allclose(mixture_weights.sum(dim=-1), torch.ones(10), atol=1e-6)
    
    def test_sampling(self):
        """Test sampling from distribution."""
        head = DistributionalHead(input_dim=512, output_dim=256)
        
        x = torch.randn(5, 512)
        mean, variance, _ = head(x)
        
        samples = head.sample(mean, variance, num_samples=10)
        
        assert samples.shape == (10, 5, 256)
    
    def test_nll_loss(self):
        """Test NLL loss computation."""
        head = DistributionalHead(input_dim=512, output_dim=256)
        
        x = torch.randn(10, 512)
        target = torch.randn(10, 256)
        
        mean, variance, _ = head(x)
        loss = head.nll_loss(mean, variance, target)
        
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)


class TestEnsembleHead:
    """Test ensemble head for epistemic uncertainty."""
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        ensemble = EnsembleHead(input_dim=512, output_dim=256, num_heads=5)
        
        x = torch.randn(10, 512)
        ensemble_mean, aleatoric_var, epistemic_var = ensemble(x)
        
        assert ensemble_mean.shape == (10, 256)
        assert aleatoric_var.shape == (10, 256)
        assert epistemic_var.shape == (10, 256)
        
        # Total uncertainty
        total_uncertainty = ensemble.total_uncertainty(aleatoric_var, epistemic_var)
        assert total_uncertainty.shape == (10, 256)


class TestActionConditionedGJEPA:
    """Test action-conditioned world model."""
    
    @pytest.fixture
    def model(self):
        return create_gjepa_model(
            latent_dim=256,
            action_dim=256,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            action_conditioned=True,
            use_distributional=True,
        )
    
    def test_forward_pass(self, model):
        """Test forward pass with actions."""
        h_seq = torch.randn(4, 10, 256)  # [batch=4, time=10, latent=256]
        actions = torch.randint(0, 51, (4, 9))  # [batch=4, time=9]
        
        mean, targets = model(h_seq, actions)
        
        assert mean.shape == (4, 9, 256)
        assert targets.shape == (4, 9, 256)
    
    def test_distributional_forward(self, model):
        """Test distributional forward pass."""
        h_seq = torch.randn(4, 10, 256)
        actions = torch.randint(0, 51, (4, 9))
        
        mean, variance, targets = model(h_seq, actions, return_distribution=True)
        
        assert mean.shape == (4, 9, 256)
        assert variance.shape == (4, 9, 256)
        assert (variance > 0).all()
    
    def test_rollout(self, model):
        """Test multi-step rollout."""
        initial_state = torch.randn(256)
        actions = torch.randint(0, 51, (10,))
        
        trajectory = model.rollout(
            initial_state=initial_state,
            actions=actions,
            horizon=10,
        )
        
        assert trajectory.shape == (11, 256)  # initial + 10 steps
    
    def test_stochastic_rollout(self, model):
        """Test stochastic rollout."""
        initial_state = torch.randn(256)
        actions = torch.randint(0, 51, (5,))
        
        trajectory = model.rollout(
            initial_state=initial_state,
            actions=actions,
            horizon=5,
            stochastic=True,
        )
        
        assert trajectory.shape == (6, 256)
    
    def test_planning(self, model):
        """Test MPC planning."""
        current_state = torch.randn(256)
        goal_state = torch.randn(256)
        
        action_sequence, cost = model.plan(
            current_state=current_state,
            goal_state=goal_state,
            horizon=5,
            num_candidates=10,
            num_iterations=2,
        )
        
        assert action_sequence.shape == (5,)
        assert isinstance(cost, float)
    
    def test_loss_computation(self, model):
        """Test loss computation."""
        h_seq = torch.randn(4, 10, 256)
        actions = torch.randint(0, 51, (4, 9))
        
        loss = model.compute_loss(h_seq, actions)
        
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestCausalReasoner:
    """Test causal reasoning module."""
    
    @pytest.fixture
    def world_model(self):
        return create_gjepa_model(
            latent_dim=256,
            action_conditioned=True,
        )
    
    @pytest.fixture
    def reasoner(self, world_model):
        return CausalReasoner(world_model=world_model)
    
    def test_interventional_prediction(self, reasoner):
        """Test interventional prediction."""
        state = torch.randn(256)
        action = torch.tensor([TheoremType.PYTHAGOREAN.value])
        
        next_state = reasoner.intervene(state, action)
        
        assert next_state.shape == (1, 256)
    
    def test_counterfactual_reasoning(self, reasoner):
        """Test counterfactual inference."""
        actual_state = torch.randn(256)
        actual_action = torch.tensor([5])
        actual_next_state = torch.randn(256)
        counterfactual_action = torch.tensor([10])
        
        cf_next_state, residual = reasoner.counterfactual(
            actual_state=actual_state,
            actual_action=actual_action,
            actual_next_state=actual_next_state,
            counterfactual_action=counterfactual_action,
        )
        
        assert cf_next_state.shape == actual_next_state.shape
        assert residual.shape == actual_next_state.shape
    
    def test_causal_graph_learning(self, reasoner):
        """Test causal graph learning."""
        states = torch.randn(20, 256)
        actions = torch.randint(0, 51, (20,))
        next_states = torch.randn(20, 256)
        
        loss = reasoner.learn_causal_graph(states, actions, next_states)
        
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestMPCPlanner:
    """Test model-predictive control planner."""
    
    @pytest.fixture
    def planner(self):
        world_model = create_gjepa_model(
            latent_dim=256,
            action_conditioned=True,
            use_distributional=True,
        )
        return MPCPlanner(
            world_model=world_model,
            horizon=5,
            num_candidates=10,
            num_iterations=2,
        )
    
    def test_planning(self, planner):
        """Test MPC planning."""
        current_state = torch.randn(256)
        goal_state = torch.randn(256)
        
        action_sequence, cost, info = planner.plan(
            current_state=current_state,
            goal_state=goal_state,
        )
        
        assert action_sequence.shape == (5,)
        assert isinstance(cost, float)
        assert 'cost_history' in info
        assert len(info['cost_history']) == 2
    
    def test_uncertainty_aware_planning(self, planner):
        """Test planning with uncertainty avoidance."""
        current_state = torch.randn(256)
        goal_state = torch.randn(256)
        
        action_sequence, cost, info = planner.plan(
            current_state=current_state,
            goal_state=goal_state,
            avoid_uncertainty=True,
        )
        
        assert action_sequence.shape == (5,)


if __name__ == "__main__":
    print("=" * 60)
    print("G-JEPA World Model Comprehensive Tests")
    print("=" * 60)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
