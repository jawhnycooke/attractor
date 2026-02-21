"""Attractor Pipeline Engine - DOT-based DAG orchestration.

Provides a graph-based execution engine that parses pipeline definitions
from GraphViz DOT files and executes them through pluggable node handlers
with condition-based routing, checkpointing, and human-in-the-loop gates.
"""

from attractor.pipeline.conditions import evaluate_condition
from attractor.pipeline.engine import PipelineEngine
from attractor.pipeline.goals import GoalGate
from attractor.pipeline.handlers import HandlerRegistry, create_default_registry
from attractor.pipeline.interviewer import CLIInterviewer, QueueInterviewer
from attractor.pipeline.models import (
    Checkpoint,
    NodeResult,
    Pipeline,
    PipelineContext,
    PipelineEdge,
    PipelineNode,
)
from attractor.pipeline.parser import parse_dot_file, parse_dot_string
from attractor.pipeline.stylesheet import ModelStylesheet, apply_stylesheet
from attractor.pipeline.validator import (
    ValidationException,
    validate_or_raise,
    validate_pipeline,
)

__all__ = [
    "Checkpoint",
    "CLIInterviewer",
    "GoalGate",
    "HandlerRegistry",
    "ModelStylesheet",
    "NodeResult",
    "Pipeline",
    "PipelineContext",
    "PipelineEdge",
    "PipelineEngine",
    "PipelineNode",
    "QueueInterviewer",
    "ValidationException",
    "apply_stylesheet",
    "create_default_registry",
    "evaluate_condition",
    "parse_dot_file",
    "parse_dot_string",
    "validate_or_raise",
    "validate_pipeline",
]
