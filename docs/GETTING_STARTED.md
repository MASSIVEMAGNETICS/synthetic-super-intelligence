# Getting Started with SSI Codex

This guide will help you get started with the Synthetic Super Intelligence Research System.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MASSIVEMAGNETICS/synthetic-super-intelligence.git
   cd synthetic-super-intelligence
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

4. **Verify installation**
   ```bash
   ssi-codex --help
   ```

## Your First Steps

### 1. Add Your First Paper

```bash
ssi-codex add \
  --title "Artificial General Intelligence: A Gentle Introduction" \
  --authors "Jane Doe, John Smith" \
  --abstract "This paper provides an overview of AGI research..." \
  --year 2024 \
  --keywords "AGI, artificial intelligence, safety"
```

### 2. List Your Papers

```bash
ssi-codex list
```

You should see your newly added paper along with the example paper included in the system.

### 3. Build the Knowledge Graph

```bash
ssi-codex build-graph
```

This will analyze all papers and create a knowledge graph connecting the concepts.

### 4. Query Concepts

```bash
ssi-codex query --concept "transformer"
```

This will show you:
- Related concepts
- Papers associated with this concept
- Relationship information

## Python API

You can also use SSI Codex programmatically:

```python
from ssi_codex import ResearchPaper, KnowledgeGraph, ConceptExtractor

# Create a paper
paper = ResearchPaper(
    title="Your Research Paper",
    authors=["Author 1", "Author 2"],
    abstract="Your abstract here...",
    year=2024,
    keywords=["keyword1", "keyword2"]
)

# Extract concepts
extractor = ConceptExtractor()
paper.concepts = extractor.extract_from_paper(paper)

# Save the paper
paper.save("research/papers")

# Work with knowledge graph
kg = KnowledgeGraph.load("data/knowledge_graph.json")
related = kg.get_related_concepts("your_concept")
```

## Common Workflows

### Adding Multiple Papers

Create a Python script:

```python
from ssi_codex import ResearchPaper, ConceptExtractor

papers_data = [
    {
        "title": "Paper 1",
        "authors": ["Author A"],
        "abstract": "Abstract 1...",
        "year": 2023
    },
    {
        "title": "Paper 2",
        "authors": ["Author B"],
        "abstract": "Abstract 2...",
        "year": 2024
    }
]

extractor = ConceptExtractor()

for data in papers_data:
    paper = ResearchPaper(**data)
    paper.concepts = extractor.extract_from_paper(paper)
    paper.save("research/papers")
```

### Analyzing Your Research Collection

```python
from pathlib import Path
from ssi_codex import ResearchPaper, KnowledgeGraph

# Load all papers
papers_dir = Path("research/papers")
papers = [
    ResearchPaper.load(str(f)) 
    for f in papers_dir.glob("*.json")
]

# Build knowledge graph
kg = KnowledgeGraph()
for paper in papers:
    for concept in paper.concepts:
        kg.add_concept(concept)
        kg.link_paper_to_concept(concept, paper.title)

# Analyze
central_concepts = kg.get_central_concepts(top_n=10)
print("Most important concepts:")
for concept, score in central_concepts:
    print(f"  {concept}: {score:.3f}")
```

### Exploring Concept Networks

```python
from ssi_codex import KnowledgeGraph

kg = KnowledgeGraph.load("data/knowledge_graph.json")

# Find concepts related to AI safety
related = kg.get_related_concepts("ai safety")
print(f"Related concepts: {related}")

# Find papers on a topic
papers = kg.get_papers_for_concept("transformer")
print(f"Papers on transformers: {len(papers)}")

# Find concept clusters
clusters = kg.find_concept_clusters()
print(f"Found {len(clusters)} concept clusters")
```

## Next Steps

1. **Add your research papers** - Start building your collection
2. **Take notes** - Use the `research/notes/` directory
3. **Define concepts** - Document key concepts in `research/concepts/`
4. **Explore the graph** - Query and analyze concept relationships
5. **Customize** - Edit `config/config.yaml` to fit your needs

## Tips

- Use descriptive keywords when adding papers
- Regularly rebuild the knowledge graph as you add papers
- Take advantage of the Python API for batch operations
- Keep notes alongside your papers for better context
- Review the most central concepts to understand your research focus

## Troubleshooting

**Problem**: CLI commands not found
- **Solution**: Make sure you installed the package with `pip install -e .`

**Problem**: Knowledge graph is empty
- **Solution**: Run `ssi-codex build-graph` after adding papers

**Problem**: Concepts not being extracted
- **Solution**: Make sure your paper has a good abstract and keywords

## Getting Help

- Check the main README.md for full documentation
- Review the example paper in `research/papers/`
- Look at the test files in `tests/` for usage examples

Happy researching!
