"""Concept extraction from research papers."""

from typing import List, Dict, Set
import re
from collections import Counter


class ConceptExtractor:
    """Extract and identify key concepts from research text."""
    
    def __init__(self):
        """Initialize the concept extractor."""
        # Common SSI-related concepts
        self.domain_keywords = {
            'artificial general intelligence', 'agi', 'superintelligence',
            'machine learning', 'deep learning', 'neural networks',
            'reinforcement learning', 'transfer learning',
            'natural language processing', 'nlp', 'computer vision',
            'cognitive architecture', 'reasoning', 'planning',
            'knowledge representation', 'ontology', 'semantic networks',
            'consciousness', 'sentience', 'self-awareness',
            'alignment', 'ai safety', 'value alignment',
            'ethical ai', 'ai governance', 'existential risk',
            'recursive self-improvement', 'intelligence explosion',
            'optimization', 'objective function', 'reward function',
            'multi-agent systems', 'game theory', 'cooperation',
            'scaling laws', 'emergent behavior', 'capabilities',
            'interpretability', 'explainability', 'transparency',
            'training', 'fine-tuning', 'prompt engineering',
            'transformer', 'attention mechanism', 'architecture',
            'benchmark', 'evaluation', 'metrics'
        }
    
    def extract_from_text(self, text: str, 
                         min_frequency: int = 2) -> List[str]:
        """Extract concepts from text based on keyword matching."""
        text_lower = text.lower()
        
        # Find all domain keywords in the text
        found_concepts = []
        for keyword in self.domain_keywords:
            if keyword in text_lower:
                found_concepts.append(keyword)
        
        # Extract potential multi-word concepts (2-4 words)
        words = re.findall(r'\b[a-z]+\b', text_lower)
        phrases = []
        
        for i in range(len(words)):
            # 2-word phrases
            if i < len(words) - 1:
                phrase = f"{words[i]} {words[i+1]}"
                phrases.append(phrase)
            # 3-word phrases
            if i < len(words) - 2:
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Add frequent phrases as concepts
        for phrase, count in phrase_counts.items():
            if count >= min_frequency and len(phrase) > 5:
                # Check if it's not too common
                common_words = {'the', 'and', 'for', 'with', 'this', 'that'}
                phrase_words = set(phrase.split())
                if not phrase_words.issubset(common_words):
                    found_concepts.append(phrase)
        
        return list(set(found_concepts))
    
    def extract_from_paper(self, paper) -> List[str]:
        """Extract concepts from a ResearchPaper object."""
        # Combine title, abstract, and keywords
        text = f"{paper.title} {paper.abstract}"
        if paper.keywords:
            text += " " + " ".join(paper.keywords)
        
        concepts = self.extract_from_text(text)
        
        # Also include explicit keywords
        if paper.keywords:
            concepts.extend([kw.lower() for kw in paper.keywords])
        
        return list(set(concepts))
    
    def identify_relationships(self, concepts: List[str]) -> List[tuple]:
        """Identify potential relationships between concepts."""
        relationships = []
        
        # Simple heuristics for relationships
        relationship_patterns = {
            'enables': ['enables', 'allows', 'supports', 'facilitates'],
            'requires': ['requires', 'needs', 'depends on'],
            'improves': ['improves', 'enhances', 'optimizes'],
            'relates_to': ['related to', 'connected to', 'associated with']
        }
        
        # This is a placeholder - in a real system, you'd use NLP
        # For now, just create generic relationships between co-occurring concepts
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                relationships.append((concept1, concept2, 'co-occurs', 1.0))
        
        return relationships
    
    def merge_similar_concepts(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Group similar concepts together."""
        # Simple similarity based on shared words
        concept_groups = {}
        
        for concept in concepts:
            words = set(concept.split())
            
            # Find existing group with shared words
            found_group = False
            for key, group in concept_groups.items():
                key_words = set(key.split())
                if len(words & key_words) > 0:
                    group.append(concept)
                    found_group = True
                    break
            
            if not found_group:
                concept_groups[concept] = [concept]
        
        return concept_groups
