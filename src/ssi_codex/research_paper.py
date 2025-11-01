"""Research Paper representation and management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import json
import os


@dataclass
class ResearchPaper:
    """Represents a research paper in the SSI Codex system."""
    
    title: str
    authors: List[str]
    abstract: str
    year: int
    url: Optional[str] = None
    doi: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    notes: str = ""
    added_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert paper to dictionary format."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "url": self.url,
            "doi": self.doi,
            "keywords": self.keywords,
            "concepts": self.concepts,
            "citations": self.citations,
            "notes": self.notes,
            "added_date": self.added_date.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ResearchPaper":
        """Create paper from dictionary."""
        if "added_date" in data and isinstance(data["added_date"], str):
            data["added_date"] = datetime.fromisoformat(data["added_date"])
        return cls(**data)
    
    def save(self, directory: str) -> str:
        """Save paper to JSON file."""
        os.makedirs(directory, exist_ok=True)
        filename = f"{self.year}_{self.title[:50].replace(' ', '_').replace('/', '_')}.json"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "ResearchPaper":
        """Load paper from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
