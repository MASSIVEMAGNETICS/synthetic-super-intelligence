"""SSI Codex - Synthetic Super Intelligence Research System.

A comprehensive system for organizing, analyzing, and synthesizing
research related to synthetic super intelligence.
"""

__version__ = "0.2.0"
__author__ = "MASSIVEMAGNETICS"

from .knowledge_graph import KnowledgeGraph
from .research_paper import ResearchPaper
from .concept_extractor import ConceptExtractor

# ZPUE - Zero-Point Understanding Engine (basic version)
from .zpue import ZeroPointUnderstandingEngine

# Enhanced ZPUE and other advanced features (require numpy)
try:
    from .zpue_enhanced import EnhancedZPUE, SemanticField
    HAS_ZPUE_ENHANCED = True
except ImportError:
    HAS_ZPUE_ENHANCED = False

# Optional import for Bando Fractal Model
try:
    from .bando_fractal_model import BandoSuperFractalLanguageModel
    HAS_FRACTAL_MODEL = True
except ImportError:
    HAS_FRACTAL_MODEL = False

# Build exports list
__all__ = [
    "KnowledgeGraph",
    "ResearchPaper",
    "ConceptExtractor",
    "ZeroPointUnderstandingEngine",
]

if HAS_ZPUE_ENHANCED:
    __all__.extend(["EnhancedZPUE", "SemanticField"])

if HAS_FRACTAL_MODEL:
    __all__.append("BandoSuperFractalLanguageModel")

# Warnings for missing optional components
import warnings
if not HAS_ZPUE_ENHANCED:
    warnings.warn("EnhancedZPUE not available: requires numpy")
if not HAS_FRACTAL_MODEL:
    warnings.warn("BandoSuperFractalLanguageModel not available: requires numpy")
