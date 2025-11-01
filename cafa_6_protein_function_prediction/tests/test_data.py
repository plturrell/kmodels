"""Unit tests for data loading and processing."""

import pytest
import numpy as np
from pathlib import Path

from src.data import (
    ProteinSample,
    build_samples,
    train_val_split,
    AMINO_ACIDS,
)
from src.data.augmentation import ProteinAugmenter


@pytest.fixture
def sample_proteins():
    """Create sample protein data for testing."""
    return [
        ProteinSample(
            accession="P12345",
            sequence="ACDEFGHIKLMNPQRSTVWY",
            go_terms=("GO:0001", "GO:0002", "GO:0003"),
        ),
        ProteinSample(
            accession="P67890",
            sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
            go_terms=("GO:0004", "GO:0005"),
        ),
        ProteinSample(
            accession="Q11111",
            sequence="GATTACA",
            go_terms=("GO:0001", "GO:0004", "GO:0006", "GO:0007"),
        ),
    ]


class TestProteinSample:
    """Test ProteinSample dataclass."""
    
    def test_creation(self):
        """Test creating a protein sample."""
        sample = ProteinSample(
            accession="TEST",
            sequence="ACDEFG",
            go_terms=("GO:0001",),
        )
        assert sample.accession == "TEST"
        assert sample.sequence == "ACDEFG"
        assert len(sample.go_terms) == 1
    
    def test_sequence_cleaning(self):
        """Test that sequences are cleaned."""
        sample = ProteinSample(
            accession="TEST",
            sequence="  ACDEFG\n",
            go_terms=(),
        )
        assert sample.sequence == "ACDEFG"
    
    def test_invalid_amino_acids(self):
        """Test handling of non-standard amino acids."""
        sample = ProteinSample(
            accession="TEST",
            sequence="ACDEFGXYZ",  # X, Y, Z not standard
            go_terms=(),
        )
        assert "contains_non_standard" in sample.metadata


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_build_samples(self, sample_proteins):
        """Test building samples from sequences and annotations."""
        sequences = {s.accession: s.sequence for s in sample_proteins}
        annotations = {s.accession: s.go_terms for s in sample_proteins}
        
        samples = build_samples(sequences, annotations)
        
        assert len(samples) == 3
        assert all(isinstance(s, ProteinSample) for s in samples)
    
    def test_train_val_split(self, sample_proteins):
        """Test train/validation splitting."""
        train, val = train_val_split(
            sample_proteins,
            val_fraction=0.33,
            seed=42,
        )
        
        assert len(train) + len(val) == len(sample_proteins)
        assert len(val) >= 1
        
        # Check no overlap
        train_ids = {s.accession for s in train}
        val_ids = {s.accession for s in val}
        assert len(train_ids & val_ids) == 0
    
    def test_stratified_split(self, sample_proteins):
        """Test stratified splitting maintains label distribution."""
        # Create more samples with varying GO term counts
        samples = sample_proteins * 10
        
        train, val = train_val_split(
            samples,
            val_fraction=0.2,
            seed=42,
            stratify=True,
        )
        
        # Check distributions are similar
        train_avg = np.mean([len(s.go_terms) for s in train])
        val_avg = np.mean([len(s.go_terms) for s in val])
        
        assert abs(train_avg - val_avg) < 1.0  # Should be close


class TestAugmentation:
    """Test data augmentation."""
    
    def test_mutation(self):
        """Test sequence mutation."""
        augmenter = ProteinAugmenter(mutation_rate=0.5, crop_prob=0.0, random_seed=42)
        
        original = "ACDEFGHIKLMNPQRSTVWY"
        mutated = augmenter.mutate_sequence(original)
        
        # Should have some mutations
        assert mutated != original
        # Should have same length
        assert len(mutated) == len(original)
        # Should only contain valid amino acids
        assert all(aa in AMINO_ACIDS for aa in mutated)
    
    def test_crop(self):
        """Test sequence cropping."""
        augmenter = ProteinAugmenter(mutation_rate=0.0, crop_prob=1.0, random_seed=42)
        
        original = "ACDEFGHIKLMNPQRSTVWY"
        cropped = augmenter.crop_sequence(original)
        
        # Should be shorter or equal
        assert len(cropped) <= len(original)
        # Should be substring
        assert cropped in original
    
    def test_augment_sample(self, sample_proteins):
        """Test augmenting a protein sample."""
        augmenter = ProteinAugmenter(mutation_rate=0.1, crop_prob=0.5, random_seed=42)
        
        original = sample_proteins[0]
        augmented = augmenter.augment_sample(original)
        
        # Should keep same accession and GO terms
        assert augmented.accession == original.accession
        assert augmented.go_terms == original.go_terms
        # Sequence may be different
        assert isinstance(augmented.sequence, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

