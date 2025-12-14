#!/usr/bin/env python3
"""
Test device consistency for G-JEPA SceneEncoder.

Verifies that SceneEncoder correctly handles CPU, CUDA, and MPS devices.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest

from src.geometry.scene_encoder import SceneEncoder
from src.geometry.scene_graph import GeometricSceneGraph
from src.geometry.scene_sequence import SceneTrace, build_scene_sequence
from src.geometry.state import State


def test_scene_encoder_cpu():
    """Test SceneEncoder on CPU device."""
    encoder = SceneEncoder(output_dim=256)
    encoder.to(torch.device("cpu"))
    
    # Create a simple scene graph
    scene = GeometricSceneGraph()
    
    # Encode
    latent = encoder.forward(scene)
    
    # Verify device
    assert latent.device.type == "cpu", f"Expected CPU device, got {latent.device}"
    assert latent.shape == (256,), f"Expected shape (256,), got {latent.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scene_encoder_cuda():
    """Test SceneEncoder on CUDA device."""
    encoder = SceneEncoder(output_dim=256)
    encoder.to(torch.device("cuda"))
    
    # Create a simple scene graph
    scene = GeometricSceneGraph()
    
    # Encode
    latent = encoder.forward(scene)
    
    # Verify device
    assert latent.device.type == "cuda", f"Expected CUDA device, got {latent.device}"
    assert latent.shape == (256,), f"Expected shape (256,), got {latent.shape}"


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available"
)
def test_scene_encoder_mps():
    """Test SceneEncoder on MPS device."""
    encoder = SceneEncoder(output_dim=256)
    encoder.to(torch.device("mps"))
    
    # Create a simple scene graph
    scene = GeometricSceneGraph()
    
    # Encode
    latent = encoder.forward(scene)
    
    # Verify device
    assert latent.device.type == "mps", f"Expected MPS device, got {latent.device}"
    assert latent.shape == (256,), f"Expected shape (256,), got {latent.shape}"


def test_scene_encoder_empty_graph_device():
    """Test that empty graph returns zero tensor on correct device."""
    for device_str in ["cpu"]:
        if device_str == "cuda" and not torch.cuda.is_available():
            continue
        if device_str == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            continue
        
        device = torch.device(device_str)
        encoder = SceneEncoder(output_dim=256)
        encoder.to(device)
        
        # Empty scene
        scene = GeometricSceneGraph()
        latent = encoder.forward(scene)
        
        assert latent.device.type == device.type, f"Expected {device.type} device, got {latent.device.type}"
        assert torch.all(latent == 0), "Empty graph should return zero vector"


def test_scene_encoder_trace_device():
    """Test that encode_trace maintains device consistency."""
    encoder = SceneEncoder(output_dim=256)
    encoder.to(torch.device("cpu"))
    
    # Create a simple trace
    scene1 = GeometricSceneGraph()
    scene2 = GeometricSceneGraph()
    state1 = State(graph=scene1)
    state2 = State(graph=scene2)
    
    trace = build_scene_sequence(
        proof_states=[state1, state2],
        trace_id="test_trace",
        problem_id="test_problem",
    )
    
    # Encode trace
    latent_seq = encoder.encode_trace(trace)
    
    # Verify device
    assert latent_seq.device.type == "cpu", f"Expected CPU device, got {latent_seq.device}"
    assert latent_seq.shape[0] == 2, f"Expected 2 timesteps, got {latent_seq.shape[0]}"
    assert latent_seq.shape[1] == 256, f"Expected latent dim 256, got {latent_seq.shape[1]}"


if __name__ == "__main__":
    print("=" * 60)
    print("Device Consistency Tests")
    print("=" * 60)
    
    print("\n1. Testing CPU device...")
    test_scene_encoder_cpu()
    print("   ✓ CPU test passed")
    
    print("\n2. Testing empty graph device...")
    test_scene_encoder_empty_graph_device()
    print("   ✓ Empty graph test passed")
    
    print("\n3. Testing trace device...")
    test_scene_encoder_trace_device()
    print("   ✓ Trace test passed")
    
    if torch.cuda.is_available():
        print("\n4. Testing CUDA device...")
        test_scene_encoder_cuda()
        print("   ✓ CUDA test passed")
    else:
        print("\n4. Skipping CUDA test (not available)")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("\n5. Testing MPS device...")
        test_scene_encoder_mps()
        print("   ✓ MPS test passed")
    else:
        print("\n5. Skipping MPS test (not available)")
    
    print("\n" + "=" * 60)
    print("All Device Tests Passed!")
    print("=" * 60)
