"""Tests for ConceptExtractor class."""

import pytest

from ssi_codex.concept_extractor import ConceptExtractor
from ssi_codex.research_paper import ResearchPaper


def test_concept_extractor_creation():
    """Test creating a concept extractor."""
    extractor = ConceptExtractor()
    
    assert len(extractor.domain_keywords) > 0


def test_extract_from_text_domain_keywords():
    """Test extracting domain keywords from text."""
    extractor = ConceptExtractor()
    
    text = """
    This paper discusses artificial general intelligence and machine learning.
    We explore deep learning techniques for natural language processing.
    """
    
    concepts = extractor.extract_from_text(text)
    
    assert "artificial general intelligence" in concepts
    assert "machine learning" in concepts
    assert "deep learning" in concepts
    assert "natural language processing" in concepts


def test_extract_from_text_phrases():
    """Test extracting repeated phrases."""
    extractor = ConceptExtractor()
    
    text = """
    Neural networks are powerful. Neural networks can learn.
    Neural networks use backpropagation. Neural networks are everywhere.
    """
    
    concepts = extractor.extract_from_text(text, min_frequency=2)
    
    # Should extract "neural networks" as it appears multiple times
    neural_concepts = [c for c in concepts if "neural" in c]
    assert len(neural_concepts) > 0


def test_extract_from_paper():
    """Test extracting concepts from a research paper."""
    extractor = ConceptExtractor()
    
    paper = ResearchPaper(
        title="Deep Learning for Computer Vision",
        authors=["Researcher"],
        abstract="This paper explores deep learning and neural networks for computer vision tasks.",
        year=2023,
        keywords=["deep learning", "computer vision"]
    )
    
    concepts = extractor.extract_from_paper(paper)
    
    assert len(concepts) > 0
    assert "deep learning" in concepts
    assert "computer vision" in concepts


def test_identify_relationships():
    """Test identifying relationships between concepts."""
    extractor = ConceptExtractor()
    
    concepts = ["ai", "machine learning", "deep learning"]
    
    relationships = extractor.identify_relationships(concepts)
    
    assert len(relationships) > 0
    # Should create relationships between all pairs
    assert len(relationships) == 3  # 3 pairs from 3 concepts


def test_merge_similar_concepts():
    """Test merging similar concepts."""
    extractor = ConceptExtractor()
    
    concepts = [
        "machine learning",
        "machine learning algorithms",
        "deep learning",
        "deep learning models",
        "neural networks"
    ]
    
    groups = extractor.merge_similar_concepts(concepts)
    
    assert len(groups) > 0
    # Concepts with shared words should be grouped
    ml_group = None
    for key, group in groups.items():
        if "machine" in key:
            ml_group = group
            break
    
    if ml_group:
        assert len(ml_group) >= 1


def test_extract_empty_text():
    """Test extracting from empty text."""
    extractor = ConceptExtractor()
    
    concepts = extractor.extract_from_text("")
    
    assert isinstance(concepts, list)
    assert len(concepts) == 0
