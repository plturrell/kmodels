# ToolOrchestra Integration

## Overview

This document describes the integration of [ToolOrchestra](https://github.com/NVlabs/ToolOrchestra) from NVIDIA Labs into the AIMO 3 competition system. ToolOrchestra is an end-to-end RL training framework for orchestrating tools and agentic workflows.

## What is ToolOrchestra?

ToolOrchestra trains small orchestrators (8B parameters) that coordinate the use of intelligent tools. It can:
- Coordinate multiple specialized models and tools
- Use reinforcement learning to optimize orchestration
- Handle multi-turn agentic tasks
- Achieve better performance than larger models while being more efficient

**Key Results from Paper:**
- Outperforms GPT-5 on HLE benchmark (37.1% vs 35.1%)
- 2.5× more efficient than GPT-5
- Uses ~30% of the cost on τ2-Bench and FRAMES

## Integration Architecture

```
┌─────────────────────────────────────┐
│   ToolOrchestraAdapter              │
│   (Orchestration Layer)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   ToolOrchestra Orchestrator        │
│   (RL-trained model)                │
└──────┬───────────┬───────────┬───────┘
       │           │           │
       ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Geometry  │ │ Algebra  │ │Number    │
│Tool      │ │ Tool     │ │Theory    │
│          │ │          │ │Tool      │
└──────────┘ └──────────┘ └──────────┘
```

## Components

### 1. ToolOrchestraAdapter (`src/orchestration/toolorchestra_adapter.py`)

Main adapter that integrates ToolOrchestra with AIMO solvers:
- Wraps AIMO domain solvers as ToolOrchestra tools
- Handles orchestration logic
- Falls back to direct solver if ToolOrchestra unavailable

### 2. AIMO Tools (`src/orchestration/aimo_tools.py`)

Wraps domain solvers as ToolOrchestra-compatible tools:
- `AIMOGeometryTool`: Geometry solver wrapper
- `AIMOAlgebraTool`: Algebra solver wrapper
- `AIMONumberTheoryTool`: Number theory solver wrapper
- `AIMOSymbolicTool`: SymPy symbolic computation

### 3. Configuration (`src/orchestration/toolorchestra_config.py`)

Configuration for ToolOrchestra integration:
- Tool definitions
- Evaluation settings
- Training configuration

## Setup

### 1. Install ToolOrchestra

```bash
cd aimo_3/vendor
git clone https://github.com/NVlabs/ToolOrchestra.git
```

### 2. Install Dependencies

```bash
# For evaluation (requires vLLM)
conda create -n vllm1 python=3.12 -y
conda activate vllm1
pip install torch transformers vllm

# For training (if training custom orchestrator)
conda create -n toolorchestra python=3.12 -y
conda activate toolorchestra
pip install -r vendor/ToolOrchestra/requirements.txt
pip install -e vendor/ToolOrchestra/training/rollout
```

### 3. Download Model Checkpoints (Optional)

```bash
# Download Nemotron-Orchestrator-8B
git clone https://huggingface.co/nvidia/Nemotron-Orchestrator-8B
export CHECKPOINT_PATH='/path/to/Nemotron-Orchestrator-8B'
```

## Usage

### Basic Usage

```python
from aimo_3.src.orchestration.toolorchestra_adapter import create_aimo_orchestrator

# Create orchestrator
orchestrator = create_aimo_orchestrator(use_toolorchestra=True)

# Solve problems (automatically orchestrates tools)
answer = orchestrator.solve("In right triangle ABC, if AC = 3 and BC = 4, find AB.")
```

### With Custom Model

```python
orchestrator = create_aimo_orchestrator(
    orchestrator_model="/path/to/custom/orchestrator",
    use_toolorchestra=True,
)
```

### Direct Tool Access

```python
from aimo_3.src.orchestration.aimo_tools import get_aimo_tools

tools = get_aimo_tools()

# Use geometry tool directly
geometry_result = tools["geometry_solver"]("Find the area of triangle ABC with base 5 and height 3")

# Use algebra tool directly
algebra_result = tools["algebra_solver"]("Solve for x: 2x + 5 = 13")
```

## Integration with Inference Pipeline

The ToolOrchestraAdapter is automatically used as the default solver in the inference pipeline:

```python
from aimo_3.src.training.inference import InferencePipeline

# Automatically uses ToolOrchestraAdapter if available
pipeline = InferencePipeline()
answers = pipeline.run_evaluation()
```

## Training Custom Orchestrator

To train a custom orchestrator optimized for AIMO problems:

```python
# Use ToolOrchestra training framework
cd vendor/ToolOrchestra/training
python resume_h100.py

# Configure for AIMO:
# - Set tool_config to use AIMO tools
# - Use AIMO problem dataset
# - Optimize for accuracy and efficiency
```

## Benefits for AIMO 3

1. **Intelligent Tool Selection**: RL-trained orchestrator learns which solver to use for each problem type
2. **Multi-Step Reasoning**: Can chain multiple tools for complex problems
3. **Efficiency**: Small orchestrator (8B) coordinates larger specialized models
4. **Adaptability**: Can learn optimal strategies from problem-solving experience
5. **Extensibility**: Easy to add new tools (combinatorics, analysis, etc.)

## Fallback Behavior

If ToolOrchestra is not available or fails:
- Falls back to `UnifiedSolver` (direct domain routing)
- Falls back to `GeometrySolver` (geometry-only)
- Falls back to `HybridSolver` (LLM + symbolic)

## Configuration

Edit `src/orchestration/toolorchestra_config.py` to:
- Add new tools
- Adjust orchestration parameters
- Configure training settings
- Set model paths

## References

- **Paper**: [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://arxiv.org/abs/2511.21689)
- **GitHub**: https://github.com/NVlabs/ToolOrchestra
- **Website**: https://research.nvidia.com/labs/lpr/ToolOrchestra/

## Status

- ✅ ToolOrchestra integrated as vendor dependency
- ✅ AIMO tools wrapped for ToolOrchestra
- ✅ Adapter layer implemented
- ✅ Integration with inference pipeline
- ⚠️ Full RL training setup (requires GPU cluster)
- ⚠️ Model checkpoint download (optional, can use fallback)

## Next Steps

1. **Train Custom Orchestrator**: Train on AIMO problem dataset
2. **Add More Tools**: Integrate combinatorics, analysis solvers
3. **Optimize**: Fine-tune orchestration for AIMO-specific patterns
4. **Evaluate**: Benchmark against direct solver approaches

