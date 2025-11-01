"""Tests for ResearchPaper class."""

import pytest
from datetime import datetime
import tempfile
import os
from pathlib import Path

from ssi_codex.research_paper import ResearchPaper


def test_research_paper_creation():
    """Test creating a research paper."""
    paper = ResearchPaper(
        title="Test Paper",
        authors=["Author One", "Author Two"],
        abstract="This is a test abstract.",
        year=2023,
        keywords=["test", "paper"]
    )
    
    assert paper.title == "Test Paper"
    assert len(paper.authors) == 2
    assert paper.year == 2023
    assert len(paper.keywords) == 2


def test_paper_to_dict():
    """Test converting paper to dictionary."""
    paper = ResearchPaper(
        title="Test Paper",
        authors=["Author One"],
        abstract="Abstract text",
        year=2023
    )
    
    data = paper.to_dict()
    
    assert data["title"] == "Test Paper"
    assert data["authors"] == ["Author One"]
    assert data["year"] == 2023
    assert "added_date" in data


def test_paper_from_dict():
    """Test creating paper from dictionary."""
    data = {
        "title": "Test Paper",
        "authors": ["Author One"],
        "abstract": "Abstract text",
        "year": 2023,
        "url": None,
        "doi": None,
        "keywords": [],
        "concepts": [],
        "citations": [],
        "notes": "",
        "added_date": datetime.now().isoformat()
    }
    
    paper = ResearchPaper.from_dict(data)
    
    assert paper.title == "Test Paper"
    assert paper.year == 2023


def test_paper_save_load():
    """Test saving and loading a paper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paper = ResearchPaper(
            title="Temporary Test Paper",
            authors=["Test Author"],
            abstract="Test abstract",
            year=2023
        )
        
        filepath = paper.save(tmpdir)
        assert os.path.exists(filepath)
        
        loaded_paper = ResearchPaper.load(filepath)
        assert loaded_paper.title == paper.title
        assert loaded_paper.authors == paper.authors
        assert loaded_paper.year == paper.year


def test_paper_with_concepts():
    """Test paper with extracted concepts."""
    paper = ResearchPaper(
        title="AI Research",
        authors=["Researcher"],
        abstract="Research on artificial intelligence",
        year=2023,
        concepts=["artificial intelligence", "machine learning"]
    )
    
    assert len(paper.concepts) == 2
    assert "artificial intelligence" in paper.concepts
