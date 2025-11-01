"""
Data augmentation strategies for protein sequences.

Implements various augmentation techniques to improve model robustness.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Sequence

import numpy as np

from .dataset import AMINO_ACIDS, ProteinSample

LOGGER = logging.getLogger(__name__)

# Amino acid substitution groups (similar properties)
SUBSTITUTION_GROUPS = [
    set("ILMV"),  # Hydrophobic aliphatic
    set("FWY"),   # Aromatic
    set("KRH"),   # Positively charged
    set("DE"),    # Negatively charged
    set("STNQ"),  # Polar uncharged
    set("AG"),    # Small
    set("P"),     # Proline (unique)
    set("C"),     # Cysteine (unique)
]


class ProteinAugmenter:
    """Augment protein sequences for training."""
    
    def __init__(
        self,
        mutation_rate: float = 0.05,
        crop_prob: float = 0.3,
        crop_ratio: tuple[float, float] = (0.8, 1.0),
        random_seed: Optional[int] = None,
    ):
        """Initialize augmenter.
        
        Args:
            mutation_rate: Probability of mutating each amino acid
            crop_prob: Probability of cropping sequence
            crop_ratio: Min/max ratio of sequence to keep when cropping
            random_seed: Random seed for reproducibility
        """
        self.mutation_rate = mutation_rate
        self.crop_prob = crop_prob
        self.crop_ratio = crop_ratio
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Build substitution map
        self._build_substitution_map()
    
    def _build_substitution_map(self):
        """Build amino acid substitution map."""
        self.substitution_map = {}
        for group in SUBSTITUTION_GROUPS:
            for aa in group:
                self.substitution_map[aa] = list(group - {aa})
    
    def mutate_sequence(self, sequence: str) -> str:
        """Apply random mutations to sequence.
        
        Args:
            sequence: Protein sequence
        
        Returns:
            Mutated sequence
        """
        sequence_list = list(sequence)
        
        for i in range(len(sequence_list)):
            if random.random() < self.mutation_rate:
                aa = sequence_list[i]
                if aa in self.substitution_map and self.substitution_map[aa]:
                    # Substitute with similar amino acid
                    sequence_list[i] = random.choice(self.substitution_map[aa])
        
        return ''.join(sequence_list)
    
    def crop_sequence(self, sequence: str) -> str:
        """Randomly crop sequence.
        
        Args:
            sequence: Protein sequence
        
        Returns:
            Cropped sequence
        """
        if random.random() > self.crop_prob:
            return sequence
        
        seq_len = len(sequence)
        crop_len = int(seq_len * random.uniform(*self.crop_ratio))
        crop_len = max(1, min(crop_len, seq_len))
        
        if crop_len >= seq_len:
            return sequence
        
        start_idx = random.randint(0, seq_len - crop_len)
        return sequence[start_idx:start_idx + crop_len]
    
    def augment_sample(self, sample: ProteinSample) -> ProteinSample:
        """Augment a protein sample.
        
        Args:
            sample: Original protein sample
        
        Returns:
            Augmented protein sample
        """
        from dataclasses import replace
        
        # Apply augmentations
        sequence = sample.sequence
        sequence = self.crop_sequence(sequence)
        sequence = self.mutate_sequence(sequence)
        
        # Create new sample
        return replace(sample, sequence=sequence)
    
    def augment_batch(
        self,
        samples: Sequence[ProteinSample],
        n_augmentations: int = 1,
    ) -> List[ProteinSample]:
        """Augment a batch of samples.
        
        Args:
            samples: Original samples
            n_augmentations: Number of augmented versions per sample
        
        Returns:
            List including original and augmented samples
        """
        augmented = list(samples)
        
        for sample in samples:
            for _ in range(n_augmentations):
                augmented.append(self.augment_sample(sample))
        
        LOGGER.info(f"Augmented {len(samples)} samples to {len(augmented)} samples")
        return augmented


def create_augmenter(config: dict) -> ProteinAugmenter:
    """Create augmenter from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        ProteinAugmenter instance
    """
    return ProteinAugmenter(
        mutation_rate=config.get('mutation_rate', 0.05),
        crop_prob=config.get('crop_prob', 0.3),
        crop_ratio=tuple(config.get('crop_ratio', [0.8, 1.0])),
        random_seed=config.get('random_seed'),
    )


__all__ = [
    "ProteinAugmenter",
    "create_augmenter",
    "SUBSTITUTION_GROUPS",
]

