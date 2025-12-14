"""Metadata schema for proof stability governance.

Implements TOON-style token representations for proof sequences
with embedded Lyapunov stability metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

from .stability_metrics import ReasoningStabilityMetrics


@dataclass
class ProofToken:
    """
    Token-Oriented representation of a proof with stability metadata.
    
    Enables stability-aware lineage tracking and governance.
    """
    
    # Core identifiers
    problem_id: str
    token_id: str = field(default_factory=lambda: f"proof_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Proof content
    proof_sequence: List[Tuple[str, Dict]] = field(default_factory=list)  # (theorem_name, match)
    answer: Optional[int] = None
    proof_found: bool = False
    
    # Stability metadata
    stability: Optional[ReasoningStabilityMetrics] = None
    
    # Provenance
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    model_version: str = "geometry_solver_v1.0"
    search_config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert token to dictionary for serialization."""
        return {
            'type': 'ProofToken',
            'id': self.token_id,
            'problem_id': self.problem_id,
            'attributes': {
                'answer': self.answer,
                'proof_found': self.proof_found,
                'proof_sequence': [
                    {'theorem': thm, 'match': match} 
                    for thm, match in self.proof_sequence
                ],
                'search_config': self.search_config
            },
            'stability': self.stability.to_dict() if self.stability else None,
            'metadata': {
                'timestamp': self.timestamp,
                'model_version': self.model_version
            }
        }
    
    def to_json(self) -> str:
        """Serialize token to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProofToken':
        """Deserialize token from dictionary."""
        # Reconstruct proof sequence
        proof_seq = [
            (step['theorem'], step['match'])
            for step in data['attributes'].get('proof_sequence', [])
        ]
        
        # Reconstruct stability metrics
        stability_data = data.get('stability')
        stability = None
        if stability_data:
            stability = ReasoningStabilityMetrics(**stability_data)
        
        return cls(
            problem_id=data['problem_id'],
            token_id=data['id'],
            proof_sequence=proof_seq,
            answer=data['attributes'].get('answer'),
            proof_found=data['attributes'].get('proof_found', False),
            stability=stability,
            timestamp=data['metadata']['timestamp'],
            model_version=data['metadata']['model_version'],
            search_config=data['attributes'].get('search_config', {})
        )


@dataclass
class StorylineToken:
    """
    Aggregate stability metrics for a collection of related proofs.
    
    Analogous to the NarrativeStoryline from the AI Nucleus paper.
    """
    
    storyline_id: str
    name: str
    description: str
    
    # Aggregate stability
    lambda_max_mean: float = 0.0
    unstable_fraction: float = 0.0
    entropy_mean: float = 0.0
    status: str = 'GREEN'  # GREEN/AMBER/RED
    
    # Linked proofs
    proof_token_ids: List[str] = field(default_factory=list)
    
    # Metadata
    horizon_days: int = 30
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    
    def to_dict(self) -> dict:
        """Convert to TOON-style dictionary."""
        return {
            'type': 'ProofStoryline',
            'id': self.storyline_id,
            'attributes': {
                'name': self.name,
                'description': self.description,
                'proof_count': len(self.proof_token_ids),
                'proof_tokens': self.proof_token_ids
            },
            'stability': {
                'lambda_max_mean': self.lambda_max_mean,
                'unstable_fraction': self.unstable_fraction,
                'entropy_mean': self.entropy_mean,
                'status': self.status,
                'horizon_days': self.horizon_days,
                'last_updated': self.last_updated
            }
        }
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)


def create_proof_token(
    problem_id: str,
    proof_sequence: List[Tuple],
    answer: Optional[int],
    stability_metrics: Optional[ReasoningStabilityMetrics],
    search_config: Dict
) -> ProofToken:
    """
    Factory function to create a ProofToken.
    
    Args:
        problem_id: Problem identifier
        proof_sequence: List of (theorem, match) tuples
        answer: Final answer
        stability_metrics: Computed stability
        search_config: Search hyperparameters used
        
    Returns:
        Initialized ProofToken
    """
    return ProofToken(
        problem_id=problem_id,
        proof_sequence=proof_sequence,
        answer=answer,
        proof_found=proof_sequence is not None and len(proof_sequence) > 0,
        stability=stability_metrics,
        search_config=search_config
    )
