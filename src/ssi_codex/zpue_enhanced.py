"""
ZPUE Enhanced Integration with SSI Codex
Semantic Field Visualization and Adaptive Learning
"""

from typing import Dict, List, Tuple, Optional
import json
import numpy as np
from pathlib import Path

from .zpue import ZeroPointUnderstandingEngine, NGramModel
from .knowledge_graph import KnowledgeGraph
from .concept_extractor import ConceptExtractor


class SemanticField:
    """Represents a continuous semantic field for ZPUE tokens."""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.token_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_history: List[Dict[str, np.ndarray]] = []
    
    def initialize_token(self, token: str) -> np.ndarray:
        """Initialize embedding for a new token."""
        if token not in self.token_embeddings:
            # Random initialization with small values
            embedding = np.random.randn(self.embedding_dim) * 0.1
            self.token_embeddings[token] = embedding
        return self.token_embeddings[token]
    
    def update_embedding(self, token: str, context_tokens: List[str], learning_rate: float = 0.01):
        """Update token embedding based on context."""
        if token not in self.token_embeddings:
            self.initialize_token(token)
        
        # Simple context-based update
        if context_tokens:
            context_embeddings = [self.initialize_token(t) for t in context_tokens if t in self.token_embeddings]
            if context_embeddings:
                context_mean = np.mean(context_embeddings, axis=0)
                # Move token embedding towards context mean
                self.token_embeddings[token] += learning_rate * (context_mean - self.token_embeddings[token])
    
    def get_similarity(self, token1: str, token2: str) -> float:
        """Compute cosine similarity between two tokens."""
        if token1 not in self.token_embeddings or token2 not in self.token_embeddings:
            return 0.0
        
        emb1 = self.token_embeddings[token1]
        emb2 = self.token_embeddings[token2]
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def find_similar_tokens(self, token: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar tokens to a given token."""
        if token not in self.token_embeddings:
            return []
        
        similarities = []
        for other_token in self.token_embeddings:
            if other_token != token:
                sim = self.get_similarity(token, other_token)
                similarities.append((other_token, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def snapshot(self):
        """Take a snapshot of current embeddings for visualization."""
        self.embedding_history.append(self.token_embeddings.copy())
    
    def get_field_vectors(self) -> Dict[str, np.ndarray]:
        """Get all token embeddings."""
        return self.token_embeddings.copy()


class EnhancedZPUE(ZeroPointUnderstandingEngine):
    """Enhanced ZPUE with semantic fields and knowledge graph integration."""
    
    def __init__(self, log_dir: Optional[str] = None, embedding_dim: int = 64):
        super().__init__(log_dir=log_dir)
        self.semantic_field = SemanticField(embedding_dim=embedding_dim)
        self.knowledge_graph = KnowledgeGraph()
        self.concept_extractor = ConceptExtractor()
        self.interaction_count = 0
    
    def update_model(self, text: str) -> None:
        """Enhanced update with semantic field and knowledge graph."""
        # Base update
        super().update_model(text)
        
        # Update semantic field
        tokens = self.tokenize(text)
        for i, token in enumerate(tokens):
            # Get context (previous and next tokens)
            context = []
            if i > 0:
                context.append(tokens[i-1])
            if i < len(tokens) - 1:
                context.append(tokens[i+1])
            
            self.semantic_field.update_embedding(token, context)
        
        # Update knowledge graph with discovered concepts
        concepts = self.concept_extractor.extract_from_text(text)
        for concept in concepts:
            self.knowledge_graph.add_concept(concept, metadata={'source': 'zpue', 'iteration': self.interaction_count})
        
        # Link related concepts
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                self.knowledge_graph.add_relationship(concept1, concept2, 'co-occurs', weight=1.0)
        
        self.interaction_count += 1
        
        # Periodic snapshot
        if self.interaction_count % 10 == 0:
            self.semantic_field.snapshot()
    
    def generate_response_with_semantics(self, length: int = 10, temperature: float = 1.0) -> Tuple[str, Dict]:
        """Generate response with semantic awareness."""
        if self.model.total_tokens == 0:
            return "I have nothing to say yet.", {}
        
        tokens: List[str] = []
        prev = self.prev_token
        semantic_info = []
        
        for _ in range(length):
            # Get next token
            token = self.model.predict_next(prev)
            tokens.append(token)
            
            # Track semantic similarity to previous
            if prev and prev in self.semantic_field.token_embeddings:
                sim = self.semantic_field.get_similarity(prev, token)
                semantic_info.append({
                    'token': token,
                    'prev': prev,
                    'similarity': float(sim)
                })
            
            prev = token
        
        response = " ".join(tokens)
        self.prev_token = tokens[-1] if tokens else None
        
        metadata = {
            'tokens': tokens,
            'semantic_flow': semantic_info,
            'vocab_size': len(self.vocab),
            'interaction_count': self.interaction_count
        }
        
        return response, metadata
    
    def get_semantic_insights(self, token: str) -> Dict:
        """Get semantic insights for a specific token."""
        if token not in self.semantic_field.token_embeddings:
            return {'error': f'Token "{token}" not in vocabulary'}
        
        similar_tokens = self.semantic_field.find_similar_tokens(token, top_k=10)
        
        # Get related concepts from knowledge graph
        related_concepts = self.knowledge_graph.get_related_concepts(token)
        
        return {
            'token': token,
            'frequency': self.vocab.get(token, 0),
            'embedding_norm': float(np.linalg.norm(self.semantic_field.token_embeddings[token])),
            'similar_tokens': [(t, float(s)) for t, s in similar_tokens],
            'related_concepts': related_concepts[:5]
        }
    
    def visualize_semantic_space(self, method: str = 'pca', n_components: int = 2) -> Dict:
        """Get semantic space visualization data."""
        embeddings = self.semantic_field.get_field_vectors()
        
        if len(embeddings) < 2:
            return {'error': 'Not enough tokens to visualize'}
        
        tokens = list(embeddings.keys())
        vectors = np.array([embeddings[t] for t in tokens])
        
        # Dimensionality reduction
        if method == 'pca' and n_components <= len(tokens):
            # Simple PCA implementation
            mean = np.mean(vectors, axis=0)
            centered = vectors - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Sort by eigenvalues
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Project
            projected = centered @ eigenvectors[:, :n_components]
            
            return {
                'method': method,
                'n_components': n_components,
                'tokens': tokens,
                'coordinates': projected.real.tolist(),
                'explained_variance': (eigenvalues[:n_components] / eigenvalues.sum()).real.tolist()
            }
        
        return {'error': f'Method "{method}" not implemented or insufficient data'}
    
    def export_state(self, filepath: str) -> None:
        """Export complete ZPUE state including semantic fields."""
        state = {
            'vocab': self.vocab,
            'unigram_counts': dict(self.model.unigram_counts),
            'bigram_counts': {str(k): v for k, v in self.model.bigram_counts.items()},
            'total_tokens': self.model.total_tokens,
            'interaction_count': self.interaction_count,
            'semantic_field_dim': self.semantic_field.embedding_dim,
            'token_embeddings': {k: v.tolist() for k, v in self.semantic_field.token_embeddings.items()},
            'knowledge_graph_size': len(self.knowledge_graph)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def interact_enhanced(self, input_text: str) -> Dict:
        """Enhanced interaction with full metadata."""
        # Update model
        self.update_model(input_text)
        
        # Generate response with semantics
        response, metadata = self.generate_response_with_semantics()
        
        # Get insights
        tokens = self.tokenize(input_text)
        if tokens:
            first_token_insights = self.get_semantic_insights(tokens[0])
        else:
            first_token_insights = {}
        
        # Log if enabled
        if self.log_dir:
            self._log_enhanced_interaction(input_text, response, metadata)
        
        return {
            'response': response,
            'metadata': metadata,
            'insights': first_token_insights,
            'stats': self.get_stats()
        }
    
    def _log_enhanced_interaction(self, user_input: str, response: str, metadata: Dict) -> None:
        """Log enhanced interaction with semantic data."""
        if not self.log_dir:
            return
        
        file_path = self.log_dir / f"zpue_enhanced_{self.interaction_count:05d}.json"
        with open(file_path, "w") as f:
            json.dump({
                "input": user_input,
                "response": response,
                "metadata": metadata,
                "timestamp": self.interaction_count
            }, f, indent=2)
