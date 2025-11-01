# Data Directory

This directory contains system data files.

## Structure

- `raw/` - Raw data files (papers, datasets, etc.)
- `processed/` - Processed data ready for analysis
- `cache/` - Temporary cache files
- `knowledge_graph.json` - The main knowledge graph

## Raw Data

Place source materials here:
- PDF files
- Text extracts
- Downloaded datasets
- External databases

## Processed Data

The system stores processed versions of data here:
- Extracted text
- Structured metadata
- Computed features

## Cache

Temporary files that can be safely deleted:
- Processing intermediates
- Download cache
- Computation results

## Knowledge Graph

The `knowledge_graph.json` file contains:
- All extracted concepts
- Relationships between concepts
- Paper-to-concept mappings
- Graph metadata

To rebuild the knowledge graph:

```bash
ssi-codex build-graph
```
