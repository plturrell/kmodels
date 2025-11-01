"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def mock_go_ontology():
    """Create a mock GO ontology for testing."""
    from src.data.go_ontology import GOOntology, GOTerm
    
    ontology = GOOntology()
    
    # Create simple hierarchy:
    # GO:0001 (root)
    #   ├── GO:0002
    #   └── GO:0003
    #       └── GO:0004
    
    ontology.add_term(GOTerm(
        id="GO:0001",
        name="biological_process",
        namespace="biological_process",
    ))
    
    ontology.add_term(GOTerm(
        id="GO:0002",
        name="child_process_1",
        namespace="biological_process",
        parents={"GO:0001"},
    ))
    
    ontology.add_term(GOTerm(
        id="GO:0003",
        name="child_process_2",
        namespace="biological_process",
        parents={"GO:0001"},
    ))
    
    ontology.add_term(GOTerm(
        id="GO:0004",
        name="grandchild_process",
        namespace="biological_process",
        parents={"GO:0003"},
    ))
    
    return ontology

