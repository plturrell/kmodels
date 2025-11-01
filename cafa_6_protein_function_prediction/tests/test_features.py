"""Unit tests for feature extraction."""

import pytest
import numpy as np

from src.data import ProteinSample
from src.features.amino_acid import extract_composition, AA_ALPHABET
from src.features.fractal_features import FractalProteinFeatures, combine_features


@pytest.fixture
def sample_protein():
    """Create a sample protein for testing."""
    return ProteinSample(
        accession="TEST",
        sequence="ACDEFGHIKLMNPQRSTVWY",
        go_terms=("GO:0001",),
    )


class TestAminoAcidFeatures:
    """Test amino acid composition features."""
    
    def test_extract_composition(self, sample_protein):
        """Test extracting amino acid composition."""
        features = extract_composition(sample_protein)
        
        assert features.accession == sample_protein.accession
        assert features.length == len(sample_protein.sequence)
        assert len(features.composition) == len(AA_ALPHABET)
        
        # Check counts sum to sequence length
        assert np.sum(features.composition) == features.length
    
    def test_normalized_composition(self, sample_protein):
        """Test normalized composition."""
        features = extract_composition(sample_protein)
        normalized = features.normalised
        
        # Should sum to 1
        assert np.isclose(np.sum(normalized), 1.0)
        # Should be in [0, 1]
        assert np.all(normalized >= 0) and np.all(normalized <= 1)


class TestFractalFeatures:
    """Test fractal feature extraction."""
    
    def test_fractal_extractor_creation(self):
        """Test creating fractal feature extractor."""
        extractor = FractalProteinFeatures(max_iter=50)
        assert extractor.max_iter == 50
    
    def test_extract_single_embedding(self):
        """Test extracting fractal features from single embedding."""
        extractor = FractalProteinFeatures(max_iter=50)
        
        # Create dummy embedding
        embedding = np.random.randn(320).astype(np.float32)
        
        features = extractor.extract_fractal_features(embedding)
        
        # Should return features
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert features.dtype == np.float32
    
    def test_extract_batch(self):
        """Test extracting fractal features from batch."""
        extractor = FractalProteinFeatures(max_iter=50)
        
        # Create dummy embeddings
        embeddings = np.random.randn(10, 320).astype(np.float32)
        
        features = extractor.extract_batch(embeddings)
        
        # Should have same number of samples
        assert features.shape[0] == embeddings.shape[0]
        # Should have features for each sample
        assert features.shape[1] > 0
    
    def test_combine_features(self):
        """Test combining original and fractal features."""
        original = np.random.randn(10, 320).astype(np.float32)
        fractal = np.random.randn(10, 100).astype(np.float32)
        
        combined = combine_features(original, fractal)
        
        # Should concatenate along feature dimension
        assert combined.shape[0] == original.shape[0]
        assert combined.shape[1] == original.shape[1] + fractal.shape[1]
    
    def test_fractal_features_deterministic(self):
        """Test that fractal features are deterministic."""
        extractor = FractalProteinFeatures(max_iter=50)
        
        embedding = np.random.randn(320).astype(np.float32)
        
        features1 = extractor.extract_fractal_features(embedding)
        features2 = extractor.extract_fractal_features(embedding)
        
        # Should be identical
        assert np.allclose(features1, features2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

