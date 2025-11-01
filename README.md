# SSI Codex - Synthetic Super Intelligence Research System

A comprehensive system for organizing, analyzing, and synthesizing research related to Synthetic Super Intelligence (SSI). This system provides tools for managing research papers, extracting concepts, building knowledge graphs, and exploring connections between ideas.

## Features

- **Research Paper Management**: Store and organize research papers with metadata
- **Concept Extraction**: Automatically extract key concepts from papers
- **Knowledge Graph**: Build and query a graph of interconnected research concepts
- **CLI Tools**: Command-line interface for all operations
- **Relationship Mapping**: Identify and track relationships between concepts
- **Bando Super Fractal Language Model**: Self-evolving AGI system with multi-paradigm optimization (requires numpy)

## Installation

```bash
# Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/synthetic-super-intelligence.git
cd synthetic-super-intelligence

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Running the Demo

```bash
python demo.py
```

This will demonstrate:
- Research paper management
- Concept extraction
- Knowledge graph building
- Bando Fractal Model integration (if numpy is available)

### Adding a Research Paper

```bash
ssi-codex add \
  --title "Attention Is All You Need" \
  --authors "Vaswani, Shazeer, Parmar" \
  --abstract "The dominant sequence transduction models..." \
  --year 2017 \
  --url "https://arxiv.org/abs/1706.03762" \
  --keywords "transformer, attention, neural networks"
```

### Listing Papers

```bash
ssi-codex list
```

### Building the Knowledge Graph

```bash
ssi-codex build-graph
```

### Querying Concepts

```bash
ssi-codex query --concept "transformer"
```

## Directory Structure

```
synthetic-super-intelligence/
├── src/ssi_codex/          # Main package code
│   ├── __init__.py
│   ├── research_paper.py   # Paper representation
│   ├── knowledge_graph.py  # Knowledge graph implementation
│   ├── concept_extractor.py # Concept extraction
│   └── cli.py              # Command-line interface
├── research/               # Research storage
│   ├── papers/            # Stored papers (JSON)
│   ├── notes/             # Research notes
│   └── concepts/          # Concept definitions
├── data/                  # Data files
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   ├── cache/            # Cache files
│   └── knowledge_graph.json # Knowledge graph
├── config/               # Configuration files
│   └── config.yaml       # System configuration
├── tests/                # Test suite
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
└── setup.py             # Package setup

```

## Core Components

### Research Paper

The `ResearchPaper` class represents a research paper with:
- Title, authors, abstract, year
- URL and DOI
- Keywords and extracted concepts
- Citations and notes
- Automatic serialization to/from JSON

### Knowledge Graph

The `KnowledgeGraph` class provides:
- Graph-based representation of concepts
- Relationship tracking between concepts
- Paper-to-concept mappings
- Centrality analysis
- Concept clustering
- Serialization to/from JSON

### Concept Extractor

The `ConceptExtractor` class offers:
- Automatic concept extraction from text
- Domain-specific keyword recognition
- Multi-word phrase detection
- Relationship identification
- Concept similarity grouping

### Bando Super Fractal Language Model (Advanced)

The `BandoSuperFractalLanguageModel` provides:
- Self-evolving code capabilities
- Multi-paradigm AI optimization (GA, NN, CA, PSO, RL)
- Recursive self-improvement mechanisms
- AST-based code mutation
- Emergent intelligence patterns

See [docs/BANDO_FRACTAL_MODEL.md](docs/BANDO_FRACTAL_MODEL.md) for detailed documentation.

## Usage Examples

### Running the Full Demo

```bash
python demo.py
```

### Python API

```python
from ssi_codex import ResearchPaper, KnowledgeGraph, ConceptExtractor

# Create a paper
paper = ResearchPaper(
    title="Deep Learning",
    authors=["Goodfellow", "Bengio", "Courville"],
    abstract="Deep learning allows computational models...",
    year=2016
)

# Extract concepts
extractor = ConceptExtractor()
concepts = extractor.extract_from_paper(paper)
paper.concepts = concepts

# Save paper
paper.save("research/papers")

# Build knowledge graph
kg = KnowledgeGraph()
for concept in paper.concepts:
    kg.add_concept(concept)
    kg.link_paper_to_concept(concept, paper.title)

# Query graph
related = kg.get_related_concepts("deep learning")
central_concepts = kg.get_central_concepts(top_n=10)

# Save graph
kg.save("data/knowledge_graph.json")
```

### Using the Bando Fractal Model

```python
from ssi_codex import BandoSuperFractalLanguageModel

# Initialize the self-evolving model
sflm = BandoSuperFractalLanguageModel()

# Process research queries
result = sflm.step(
    "What are the key challenges in AGI alignment?",
    mode="evolve"
)

print(f"Intent: {result['token_info']['intent']}")
print(f"Output: {result['transformer_mesh_output_summary']}")

# Test self-modification
result = sflm.step(
    "self.mutate this optimization algorithm",
    mode="evolve"
)

# Check if code evolved
if result['evolved_code'] != result['input']:
    print(f"New variant created: Gen{sflm.growth_engine.generation}")
```

## Configuration

Edit `config/config.yaml` to customize:
- Data directories
- Extraction parameters
- Knowledge graph settings
- Output formats

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Structure

The system is designed to be modular and extensible:
- Each component is self-contained
- Clear interfaces between modules
- Easy to add new extractors or analyzers
- Pluggable storage backends

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is part of the MASSIVEMAGNETICS research initiative.

## Future Enhancements

- [ ] Advanced NLP for better concept extraction
- [ ] PDF parsing for direct paper ingestion
- [ ] Semantic similarity search
- [ ] Citation network analysis
- [ ] Interactive visualization
- [ ] Web interface
- [ ] Export to various formats
- [ ] Integration with academic databases
- [ ] Automated paper recommendations
- [ ] Multi-language support
- [ ] Distributed fractal model evolution
- [ ] Quantum-inspired optimization
- [ ] Multi-modal processing (vision, audio)
- [ ] Real-time evolution dashboard
