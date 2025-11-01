"""Tests for KnowledgeGraph class."""

import pytest
import tempfile
import os

from ssi_codex.knowledge_graph import KnowledgeGraph


def test_knowledge_graph_creation():
    """Test creating an empty knowledge graph."""
    kg = KnowledgeGraph()
    
    assert len(kg) == 0
    assert len(kg.concept_papers) == 0


def test_add_concept():
    """Test adding concepts to the graph."""
    kg = KnowledgeGraph()
    
    kg.add_concept("artificial intelligence")
    kg.add_concept("machine learning")
    
    assert len(kg) == 2
    assert "artificial intelligence" in kg.graph
    assert "machine learning" in kg.graph


def test_add_relationship():
    """Test adding relationships between concepts."""
    kg = KnowledgeGraph()
    
    kg.add_relationship("machine learning", "deep learning", "includes")
    
    assert len(kg) == 2
    assert kg.graph.has_edge("machine learning", "deep learning")
    
    edge_data = kg.graph.edges["machine learning", "deep learning"]
    assert edge_data["relation"] == "includes"


def test_link_paper_to_concept():
    """Test linking papers to concepts."""
    kg = KnowledgeGraph()
    
    kg.link_paper_to_concept("artificial intelligence", "paper1")
    kg.link_paper_to_concept("artificial intelligence", "paper2")
    kg.link_paper_to_concept("machine learning", "paper1")
    
    ai_papers = kg.get_papers_for_concept("artificial intelligence")
    assert len(ai_papers) == 2
    assert "paper1" in ai_papers
    assert "paper2" in ai_papers
    
    ml_papers = kg.get_papers_for_concept("machine learning")
    assert len(ml_papers) == 1


def test_get_related_concepts():
    """Test finding related concepts."""
    kg = KnowledgeGraph()
    
    kg.add_relationship("ai", "ml", "includes")
    kg.add_relationship("ml", "deep learning", "includes")
    kg.add_relationship("ml", "reinforcement learning", "includes")
    
    related = kg.get_related_concepts("ml")
    
    assert "ai" in related or "deep learning" in related or "reinforcement learning" in related


def test_get_central_concepts():
    """Test getting central concepts."""
    kg = KnowledgeGraph()
    
    # Create a small network
    kg.add_relationship("ai", "ml", "includes")
    kg.add_relationship("ai", "nlp", "includes")
    kg.add_relationship("ai", "cv", "includes")
    kg.add_relationship("ml", "deep learning", "includes")
    
    central = kg.get_central_concepts(top_n=3)
    
    assert len(central) > 0
    assert central[0][0] == "ai"  # AI should be most central


def test_save_load_graph():
    """Test saving and loading knowledge graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kg = KnowledgeGraph()
        
        kg.add_relationship("concept1", "concept2", "relates_to")
        kg.link_paper_to_concept("concept1", "paper1")
        
        filepath = os.path.join(tmpdir, "test_graph.json")
        kg.save(filepath)
        
        assert os.path.exists(filepath)
        
        loaded_kg = KnowledgeGraph.load(filepath)
        
        assert len(loaded_kg) == len(kg)
        assert "concept1" in loaded_kg.graph
        assert "concept2" in loaded_kg.graph
        assert loaded_kg.graph.has_edge("concept1", "concept2")
        
        papers = loaded_kg.get_papers_for_concept("concept1")
        assert "paper1" in papers


def test_find_concept_clusters():
    """Test finding concept clusters."""
    kg = KnowledgeGraph()
    
    # Create two separate clusters
    kg.add_relationship("ai", "ml", "includes")
    kg.add_relationship("ml", "dl", "includes")
    
    kg.add_relationship("physics", "quantum", "includes")
    kg.add_relationship("quantum", "computing", "includes")
    
    clusters = kg.find_concept_clusters()
    
    assert len(clusters) == 2
