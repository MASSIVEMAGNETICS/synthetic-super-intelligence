"""Knowledge Graph for connecting research concepts."""

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    # Provide minimal fallback
    class DummyGraph:
        def __init__(self):
            self.nodes_dict = {}
            self.edges_dict = {}
        
        def has_node(self, n):
            return n in self.nodes_dict
        
        def add_node(self, n, **attrs):
            self.nodes_dict[n] = attrs
        
        def add_edge(self, u, v, **attrs):
            if u not in self.edges_dict:
                self.edges_dict[u] = {}
            self.edges_dict[u][v] = attrs
        
        def has_edge(self, u, v):
            return u in self.edges_dict and v in self.edges_dict[u]
        
        def nodes(self):
            return self.nodes_dict.keys()
        
        def edges(self):
            edges = []
            for u, targets in self.edges_dict.items():
                for v in targets:
                    edges.append((u, v))
            return edges
        
        def number_of_edges(self):
            return sum(len(targets) for targets in self.edges_dict.values())
        
        def __getitem__(self, n):
            return self.nodes_dict.get(n, {})
        
        def __len__(self):
            return len(self.nodes_dict)

from typing import Dict, List, Set, Optional, Tuple
import json
import os


class KnowledgeGraph:
    """Knowledge graph for SSI research concepts and relationships."""
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            self.graph = DummyGraph()
        self.concept_papers = {}  # Maps concepts to paper IDs
        
    def add_concept(self, concept: str, metadata: Optional[Dict] = None):
        """Add a concept node to the graph."""
        if not self.graph.has_node(concept):
            attrs = metadata or {}
            attrs['type'] = 'concept'
            self.graph.add_node(concept, **attrs)
            self.concept_papers[concept] = set()
    
    def add_relationship(self, source: str, target: str, 
                        relation_type: str, weight: float = 1.0):
        """Add a relationship between concepts."""
        self.add_concept(source)
        self.add_concept(target)
        self.graph.add_edge(source, target, 
                           relation=relation_type, 
                           weight=weight)
    
    def link_paper_to_concept(self, concept: str, paper_id: str):
        """Link a research paper to a concept."""
        if concept not in self.concept_papers:
            self.add_concept(concept)
        self.concept_papers[concept].add(paper_id)
    
    def get_related_concepts(self, concept: str, 
                            max_depth: int = 2) -> List[str]:
        """Get concepts related to the given concept."""
        if HAS_NETWORKX:
            if concept not in self.graph:
                return []
            
            related = set()
            try:
                # Get descendants (concepts this one relates to)
                descendants = nx.descendants(self.graph, concept)
                related.update(descendants)
                
                # Get predecessors (concepts that relate to this one)
                predecessors = nx.ancestors(self.graph, concept)
                related.update(predecessors)
            except:
                pass
            
            return list(related)
        else:
            # Fallback: return directly connected concepts
            related = set()
            if concept in self.graph.edges_dict:
                related.update(self.graph.edges_dict[concept].keys())
            # Check reverse edges
            for source, targets in self.graph.edges_dict.items():
                if concept in targets:
                    related.add(source)
            return list(related)
    
    def get_papers_for_concept(self, concept: str) -> Set[str]:
        """Get all papers associated with a concept."""
        return self.concept_papers.get(concept, set())
    
    def find_concept_clusters(self) -> List[Set[str]]:
        """Find clusters of related concepts."""
        if HAS_NETWORKX:
            # Convert to undirected for clustering
            undirected = self.graph.to_undirected()
            
            # Find connected components
            components = list(nx.connected_components(undirected))
            
            return components
        else:
            # Simple fallback: each concept is its own cluster
            return [{concept} for concept in self.graph.nodes()]
    
    def get_central_concepts(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the most central concepts by degree centrality."""
        if len(self.graph) == 0:
            return []
        
        if HAS_NETWORKX:
            centrality = nx.degree_centrality(self.graph)
            sorted_concepts = sorted(centrality.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
        else:
            # Fallback: count edges
            degree = {}
            for node in self.graph.nodes():
                count = 0
                if node in self.graph.edges_dict:
                    count += len(self.graph.edges_dict[node])
                # Count incoming edges
                for source, targets in self.graph.edges_dict.items():
                    if node in targets:
                        count += 1
                degree[node] = count
            
            total = max(sum(degree.values()), 1)
            centrality = {k: v/total for k, v in degree.items()}
            sorted_concepts = sorted(centrality.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)
        
        return sorted_concepts[:top_n]
    
    def save(self, filepath: str):
        """Save knowledge graph to file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        data = {
            'nodes': [
                {
                    'id': node,
                    'attrs': dict(self.graph[node] if HAS_NETWORKX else self.graph.nodes_dict.get(node, {}))
                }
                for node in self.graph.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'attrs': (dict(self.graph.edges[u, v]) if HAS_NETWORKX 
                             else self.graph.edges_dict.get(u, {}).get(v, {}))
                }
                for u, v in self.graph.edges()
            ],
            'concept_papers': {
                concept: list(papers)
                for concept, papers in self.concept_papers.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """Load knowledge graph from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        kg = cls()
        
        # Add nodes
        for node_data in data['nodes']:
            kg.graph.add_node(node_data['id'], **node_data.get('attrs', {}))
        
        # Add edges
        for edge_data in data['edges']:
            kg.graph.add_edge(
                edge_data['source'],
                edge_data['target'],
                **edge_data.get('attrs', {})
            )
        
        # Add concept-paper mappings
        kg.concept_papers = {
            concept: set(papers)
            for concept, papers in data['concept_papers'].items()
        }
        
        return kg
    
    def __len__(self) -> int:
        """Return number of concepts in the graph."""
        if HAS_NETWORKX:
            return len(self.graph)
        else:
            return len(self.graph.nodes_dict)
