"""Gene Ontology (GO) hierarchy parser and utilities.

The GO ontology is a directed acyclic graph (DAG) where terms have parent-child
relationships. This module provides utilities to parse the OBO format and work
with the hierarchy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

LOGGER = logging.getLogger(__name__)


@dataclass
class GOTerm:
    """Represents a single GO term with its metadata and relationships."""
    
    id: str
    name: str = ""
    namespace: str = ""  # biological_process, molecular_function, cellular_component
    definition: str = ""
    is_obsolete: bool = False
    parents: Set[str] = field(default_factory=set)  # is_a relationships
    children: Set[str] = field(default_factory=set)
    
    @property
    def aspect(self) -> str:
        """Return single-letter aspect code (P/F/C)."""
        if self.namespace == "biological_process":
            return "P"
        elif self.namespace == "molecular_function":
            return "F"
        elif self.namespace == "cellular_component":
            return "C"
        return ""


class GOOntology:
    """Gene Ontology hierarchy manager."""
    
    def __init__(self):
        self.terms: Dict[str, GOTerm] = {}
    
    def add_term(self, term: GOTerm) -> None:
        """Add a GO term to the ontology."""
        self.terms[term.id] = term
        
        # Update children relationships
        for parent_id in term.parents:
            if parent_id in self.terms:
                self.terms[parent_id].children.add(term.id)
    
    def get_term(self, term_id: str) -> Optional[GOTerm]:
        """Retrieve a GO term by ID."""
        return self.terms.get(term_id)
    
    def get_ancestors(self, term_id: str, include_self: bool = True) -> Set[str]:
        """Get all ancestor terms (parents, grandparents, etc.) of a term.
        
        Args:
            term_id: GO term identifier
            include_self: Whether to include the term itself in results
        
        Returns:
            Set of GO term IDs including all ancestors
        """
        ancestors = {term_id} if include_self else set()
        term = self.get_term(term_id)
        
        if term is None:
            return ancestors
        
        # Recursively collect all parents
        for parent_id in term.parents:
            ancestors.update(self.get_ancestors(parent_id, include_self=True))
        
        return ancestors
    
    def get_descendants(self, term_id: str, include_self: bool = True) -> Set[str]:
        """Get all descendant terms (children, grandchildren, etc.) of a term."""
        descendants = {term_id} if include_self else set()
        term = self.get_term(term_id)
        
        if term is None:
            return descendants
        
        # Recursively collect all children
        for child_id in term.children:
            descendants.update(self.get_descendants(child_id, include_self=True))
        
        return descendants
    
    def propagate_annotations(self, term_ids: Set[str]) -> Set[str]:
        """Propagate annotations up the hierarchy.
        
        If a protein is annotated with a term, it should also be annotated
        with all ancestor terms (true path rule).
        
        Args:
            term_ids: Set of GO term IDs
        
        Returns:
            Extended set including all ancestor terms
        """
        propagated = set()
        for term_id in term_ids:
            propagated.update(self.get_ancestors(term_id, include_self=True))
        return propagated


def parse_obo_file(obo_path: Path) -> GOOntology:
    """Parse a GO ontology OBO file.
    
    Args:
        obo_path: Path to the go-basic.obo file
    
    Returns:
        GOOntology object with parsed terms and relationships
    """
    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}")
    
    ontology = GOOntology()
    current_term: Optional[GOTerm] = None
    
    with obo_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if line == "[Term]":
                # Save previous term if exists
                if current_term is not None:
                    ontology.add_term(current_term)
                # Start new term
                current_term = GOTerm(id="")
            
            elif line.startswith("id:") and current_term is not None:
                current_term.id = line.split("id:")[1].strip()
            
            elif line.startswith("name:") and current_term is not None:
                current_term.name = line.split("name:")[1].strip()
            
            elif line.startswith("namespace:") and current_term is not None:
                current_term.namespace = line.split("namespace:")[1].strip()
            
            elif line.startswith("def:") and current_term is not None:
                current_term.definition = line.split("def:")[1].strip()
            
            elif line.startswith("is_a:") and current_term is not None:
                parent_id = line.split("is_a:")[1].split("!")[0].strip()
                current_term.parents.add(parent_id)
            
            elif line.startswith("is_obsolete:") and current_term is not None:
                current_term.is_obsolete = "true" in line.lower()
    
    # Add last term
    if current_term is not None:
        ontology.add_term(current_term)
    
    LOGGER.info("Parsed %d GO terms from %s", len(ontology.terms), obo_path)
    return ontology


__all__ = [
    "GOTerm",
    "GOOntology",
    "parse_obo_file",
]

