# Example Research Papers

This directory contains research papers in JSON format.

## Adding Papers

You can add papers using the CLI:

```bash
ssi-codex add --title "Your Paper Title" --authors "Author1, Author2" --abstract "Abstract text" --year 2023
```

Or programmatically:

```python
from ssi_codex import ResearchPaper

paper = ResearchPaper(
    title="Paper Title",
    authors=["Author1", "Author2"],
    abstract="Abstract text",
    year=2023
)
paper.save("research/papers")
```

## Format

Papers are stored as JSON files with the following structure:

```json
{
  "title": "Paper Title",
  "authors": ["Author1", "Author2"],
  "abstract": "Abstract text",
  "year": 2023,
  "url": "https://example.com/paper",
  "doi": "10.1234/example",
  "keywords": ["keyword1", "keyword2"],
  "concepts": ["concept1", "concept2"],
  "citations": ["paper1", "paper2"],
  "notes": "Personal notes",
  "added_date": "2023-01-01T00:00:00"
}
```
