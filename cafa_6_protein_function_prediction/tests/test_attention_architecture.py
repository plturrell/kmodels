"""Tests for the attention-based Lightning architecture."""

import torch

from src.training.lightning_module import ProteinLightningModule
from src.config.training import OptimizerConfig


def _build_module(architecture: str = "attention"):
    return ProteinLightningModule(
        embedding_dim=16,
        hidden_dims=(8, 4),
        dropout=0.1,
        architecture=architecture,
        attention_heads=4,
        class_names=[f"GO:{i:07d}" for i in range(4)],
        optimizer_cfg=OptimizerConfig(learning_rate=1e-3),
        val_accessions=["P12345"],
        val_ground_truth={"P12345": ["GO:0000001"]},
        ontology=None,
    )


def test_attention_forward_and_extract():
    module = _build_module("attention")
    module.eval()
    embeddings = torch.rand(2, 16)
    logits = module(embeddings)
    assert logits.shape == (2, 4)

    single_embedding = embeddings[0].unsqueeze(0)
    logits_single, attention = module.extract_attention(single_embedding)
    assert logits_single.shape == (1, 4)
    assert attention.ndim == 3
    matrix = attention.squeeze(0)
    assert matrix.shape[0] == matrix.shape[1], "Attention matrix must be square"


def test_mlp_architecture_forward():
    module = _build_module("mlp")
    embeddings = torch.rand(3, 16)
    logits = module(embeddings)
    assert logits.shape == (3, 4)
    # Calling extract_attention on MLP should raise
    try:
        module.extract_attention(embeddings[:1])
    except RuntimeError:
        pass
    else:
        raise AssertionError("extract_attention should fail for non-attention architecture")
