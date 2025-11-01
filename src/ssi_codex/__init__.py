"""SSI Codex - Synthetic Super Intelligence Research System.

A comprehensive system for organizing, analyzing, and synthesizing
research related to synthetic super intelligence.
"""

__version__ = "0.1.0"
__author__ = "MASSIVEMAGNETICS"

from .knowledge_graph import KnowledgeGraph
from .research_paper import ResearchPaper
from .concept_extractor import ConceptExtractor

# Optional import for Bando Fractal Model
try:
    from .bando_fractal_model import BandoSuperFractalLanguageModel
    __all__ = [
        "KnowledgeGraph",
        "ResearchPaper",
        "ConceptExtractor",
        "BandoSuperFractalLanguageModel",
    ]
except ImportError as e:
    # Fractal model requires numpy, torch (optional)
    __all__ = [
        "KnowledgeGraph",
        "ResearchPaper",
        "ConceptExtractor",
    ]
    import warnings
    warnings.warn(f"BandoSuperFractalLanguageModel not available: {e}")
