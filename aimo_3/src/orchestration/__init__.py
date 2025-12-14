"""ToolOrchestra integration for AIMO problem solving."""

from .toolorchestra_adapter import ToolOrchestraAdapter, create_aimo_orchestrator
from .aimo_tools import (
    AIMOGeometryTool,
    AIMOAlgebraTool,
    AIMONumberTheoryTool,
    AIMOCombinatoricsTool,
    AIMOGraphTool,
    AIMOAnalysisTool,
)

try:
    from .stability_tracker import (
        StabilityTracker,
        StabilityStatus,
        ProofToken,
        OrchestrationStabilityMetrics,
    )
    __all__ = [
        "ToolOrchestraAdapter",
        "create_aimo_orchestrator",
        "AIMOGeometryTool",
        "AIMOAlgebraTool",
        "AIMONumberTheoryTool",
        "AIMOCombinatoricsTool",
        "AIMOGraphTool",
        "AIMOAnalysisTool",
        "StabilityTracker",
        "StabilityStatus",
        "ProofToken",
        "OrchestrationStabilityMetrics",
    ]
except ImportError:
    __all__ = [
        "ToolOrchestraAdapter",
        "create_aimo_orchestrator",
        "AIMOGeometryTool",
        "AIMOAlgebraTool",
        "AIMONumberTheoryTool",
        "AIMOCombinatoricsTool",
        "AIMOGraphTool",
        "AIMOAnalysisTool",
    ]

