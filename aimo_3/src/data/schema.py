from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core enums
# ---------------------------------------------------------------------------


class Difficulty(str, Enum):
    TRAINING_EASY = "training_easy"
    TRAINING_MEDIUM = "training_medium"
    TRAINING_HARD = "training_hard"
    OLYMPIAD_EASY = "olympiad_easy"
    OLYMPIAD_MEDIUM = "olympiad_medium"
    OLYMPIAD_HARD = "olympiad_hard"


class SourceType(str, Enum):
    SYNTHETIC = "synthetic"
    TRANSFORMED_EXTERNAL = "transformed_external"
    PUBLIC_DOMAIN = "public_domain"
    OTHER_OPEN = "other_open"


class ReviewStatus(str, Enum):
    GENERATED = "generated"
    AUTO_VALIDATED = "auto_validated"
    MANUALLY_VERIFIED = "manually_verified"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST_LIKE = "test_like"


class TopicTag(str, Enum):
    NUMBER_THEORY = "number_theory"
    ALGEBRA = "algebra"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    INEQUALITIES = "inequalities"
    FUNCTIONAL_EQUATIONS = "functional_equations"
    POLYNOMIALS = "polynomials"
    SEQUENCES_SERIES = "sequences_series"
    PROBABILITY = "probability"
    GRAPH_THEORY = "graph_theory"
    MISC = "misc"


class SpanStream(str, Enum):
    SYSTEM = "system"
    PROMPT = "prompt"
    COMPLETION = "completion"


class SpanKind(str, Enum):
    PROBLEM = "problem"
    SCRATCHPAD = "scratchpad"
    FINAL_ANSWER = "final_answer"
    TOOL_CALL = "tool_call"


# ---------------------------------------------------------------------------
# Token-oriented representation and shared structures
# ---------------------------------------------------------------------------


class TokenSpan(BaseModel):
    """Token-oriented view of an example for training.

    This is intentionally model-agnostic: we keep only roles, streams, and
    approximate token counts; actual tokenization happens downstream.
    """

    stream: SpanStream
    role: Literal["system", "user", "assistant"]
    kind: SpanKind
    text: str
    approxTokens: Optional[int] = None


class SolutionStep(BaseModel):
    type: str
    textLatex: str
    isKeyStep: bool = False


class ToolCall(BaseModel):
    tool: str
    description: Optional[str] = None
    code: str
    result: str


class SourceInfo(BaseModel):
    sourceType: SourceType
    reference: str
    sourceUrl: Optional[str] = None


class Owner(BaseModel):
    name: str
    email: Optional[str] = None
    affiliation: Optional[str] = None
    role: Optional[str] = None


# ---------------------------------------------------------------------------
# Main problem / corpus models
# ---------------------------------------------------------------------------


class MathProblem(BaseModel):
    """Schema for a single math problem instance (JSONL record)."""

    # Identifiers and metadata
    id: str
    fullyQualifiedName: str
    version: int = 1
    createdAt: int  # epoch ms
    updatedAt: int  # epoch ms
    name: str
    language: str = "en"

    # Problem content
    problemLatex: str
    problemPlaintext: Optional[str] = None
    answer: int = Field(..., ge=0, le=99999)
    answerRange: Dict[str, int] = Field(
        default_factory=lambda: {"min": 0, "max": 99999}
    )

    # Classification
    topicTags: List[TopicTag]
    difficulty: Difficulty
    source: SourceInfo
    license: str
    tags: List[str] = []
    domains: List[str] = []
    owners: List[Owner] = []

    # Reasoning + status
    estimatedTimeMinutes: Optional[int] = None
    reviewStatus: Optional[ReviewStatus] = None
    solutionLatex: Optional[str] = None
    solutionSteps: List[SolutionStep] = []
    toolTrace: List[ToolCall] = []

    # Training-oriented structure
    split: Split = Split.TRAIN
    tokenSpans: List[TokenSpan] = []

    # Open-ended extension area, including geometry payloads
    extension: Dict[str, Any] = Field(default_factory=dict)


class GeneratorInfo(BaseModel):
    name: str
    version: str
    repoUrl: Optional[str] = None
    commitHash: Optional[str] = None
    scriptPath: Optional[str] = None


class DataFileInfo(BaseModel):
    path: str
    split: Split
    numExamples: int


class CorpusMetadata(BaseModel):
    """Corpus-level metadata for a dataset release."""

    id: str
    name: str
    displayName: str
    description: str

    version: str
    schemaVersion: str
    createdAt: int
    updatedAt: int

    license: str
    sourceUrl: Optional[str] = None

    tags: List[str] = []
    domains: List[str] = []
    maintainers: List[Owner] = []

    stats: Dict[str, int] = {}
    generators: List[GeneratorInfo] = []
    dataFiles: List[DataFileInfo] = []

    extension: Dict[str, Any] = Field(default_factory=dict)

