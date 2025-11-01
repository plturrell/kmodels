"""
Information Content (IC) calculation for GO terms.

IC is used for semantic similarity calculations in CAFA evaluation.
IC(term) = -log(P(term)) where P(term) is the probability of the term
appearing in the annotation corpus.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

from ..data.go_ontology import GOOntology

LOGGER = logging.getLogger(__name__)


class InformationContent:
    """Calculate and store information content for GO terms."""
    
    def __init__(self, ontology: Optional[GOOntology] = None):
        """Initialize IC calculator.
        
        Args:
            ontology: GO ontology for propagating annotations
        """
        self.ontology = ontology
        self.ic_scores: Dict[str, float] = {}
        self.term_counts: Dict[str, int] = {}
        self.total_annotations = 0
    
    def compute_from_annotations(
        self,
        annotations: Dict[str, Set[str]],
        propagate: bool = True,
    ) -> None:
        """Compute IC scores from annotation data.
        
        Args:
            annotations: Dict mapping protein_id -> set of GO terms
            propagate: Whether to propagate annotations up the hierarchy
        """
        LOGGER.info("Computing information content from %d proteins", len(annotations))
        
        # Count term occurrences
        term_counter = Counter()
        
        for protein_id, terms in annotations.items():
            if propagate and self.ontology:
                # Propagate annotations up the hierarchy
                terms = self.ontology.propagate_annotations(terms)
            
            for term in terms:
                term_counter[term] += 1
        
        self.term_counts = dict(term_counter)
        self.total_annotations = sum(term_counter.values())
        
        # Calculate IC scores: IC(term) = -log(P(term))
        for term, count in self.term_counts.items():
            probability = count / self.total_annotations
            self.ic_scores[term] = -np.log(probability)
        
        LOGGER.info("Computed IC for %d unique GO terms", len(self.ic_scores))
        LOGGER.info("IC range: [%.4f, %.4f]", min(self.ic_scores.values()), max(self.ic_scores.values()))
    
    def get_ic(self, term: str, default: float = 0.0) -> float:
        """Get IC score for a term.
        
        Args:
            term: GO term ID
            default: Default value if term not found
        
        Returns:
            IC score
        """
        return self.ic_scores.get(term, default)
    
    def semantic_similarity(
        self,
        term1: str,
        term2: str,
        method: str = "resnik",
    ) -> float:
        """Calculate semantic similarity between two GO terms.
        
        Args:
            term1: First GO term
            term2: Second GO term
            method: Similarity method ('resnik', 'lin', 'jiang')
        
        Returns:
            Semantic similarity score
        """
        if not self.ontology:
            # Fallback: exact match
            return 1.0 if term1 == term2 else 0.0
        
        # Find common ancestors
        ancestors1 = self.ontology.get_ancestors(term1, include_self=True)
        ancestors2 = self.ontology.get_ancestors(term2, include_self=True)
        common_ancestors = ancestors1 & ancestors2
        
        if not common_ancestors:
            return 0.0
        
        # Find most informative common ancestor (MICA)
        mica_ic = max((self.get_ic(term) for term in common_ancestors), default=0.0)
        
        if method == "resnik":
            # Resnik similarity: IC of MICA
            return mica_ic
        
        elif method == "lin":
            # Lin similarity: 2 * IC(MICA) / (IC(term1) + IC(term2))
            ic1 = self.get_ic(term1)
            ic2 = self.get_ic(term2)
            if ic1 + ic2 == 0:
                return 0.0
            return (2 * mica_ic) / (ic1 + ic2)
        
        elif method == "jiang":
            # Jiang-Conrath distance: IC(term1) + IC(term2) - 2 * IC(MICA)
            # Convert to similarity: 1 / (1 + distance)
            ic1 = self.get_ic(term1)
            ic2 = self.get_ic(term2)
            distance = ic1 + ic2 - 2 * mica_ic
            return 1.0 / (1.0 + distance)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def save(self, path: Path) -> None:
        """Save IC scores to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'ic_scores': self.ic_scores,
                'term_counts': self.term_counts,
                'total_annotations': self.total_annotations,
            }, f, indent=2)
        LOGGER.info("Saved IC scores to %s", path)
    
    @classmethod
    def load(cls, path: Path, ontology: Optional[GOOntology] = None) -> InformationContent:
        """Load IC scores from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        ic = cls(ontology=ontology)
        ic.ic_scores = data['ic_scores']
        ic.term_counts = data['term_counts']
        ic.total_annotations = data['total_annotations']
        LOGGER.info("Loaded IC scores for %d terms from %s", len(ic.ic_scores), path)
        return ic


__all__ = [
    "InformationContent",
]

