"""Command-line interface for SSI Codex."""

import argparse
import sys
import os
from pathlib import Path

from .research_paper import ResearchPaper
from .knowledge_graph import KnowledgeGraph
from .concept_extractor import ConceptExtractor


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="SSI Codex - Synthetic Super Intelligence Research System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add paper command
    add_parser = subparsers.add_parser('add', help='Add a research paper')
    add_parser.add_argument('--title', required=True, help='Paper title')
    add_parser.add_argument('--authors', required=True, help='Comma-separated authors')
    add_parser.add_argument('--abstract', required=True, help='Paper abstract')
    add_parser.add_argument('--year', type=int, required=True, help='Publication year')
    add_parser.add_argument('--url', help='Paper URL')
    add_parser.add_argument('--keywords', help='Comma-separated keywords')
    
    # List papers command
    list_parser = subparsers.add_parser('list', help='List all papers')
    list_parser.add_argument('--directory', default='research/papers', 
                            help='Directory containing papers')
    
    # Build graph command
    graph_parser = subparsers.add_parser('build-graph', 
                                         help='Build knowledge graph from papers')
    graph_parser.add_argument('--papers-dir', default='research/papers',
                             help='Directory containing papers')
    graph_parser.add_argument('--output', default='data/knowledge_graph.json',
                             help='Output file for knowledge graph')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge graph')
    query_parser.add_argument('--concept', required=True, 
                             help='Concept to query')
    query_parser.add_argument('--graph', default='data/knowledge_graph.json',
                             help='Knowledge graph file')
    
    args = parser.parse_args()
    
    if args.command == 'add':
        add_paper(args)
    elif args.command == 'list':
        list_papers(args)
    elif args.command == 'build-graph':
        build_knowledge_graph(args)
    elif args.command == 'query':
        query_graph(args)
    else:
        parser.print_help()
        return 1
    
    return 0


def add_paper(args):
    """Add a research paper to the system."""
    authors = [a.strip() for a in args.authors.split(',')]
    keywords = [k.strip() for k in args.keywords.split(',')] if args.keywords else []
    
    paper = ResearchPaper(
        title=args.title,
        authors=authors,
        abstract=args.abstract,
        year=args.year,
        url=args.url,
        keywords=keywords
    )
    
    # Extract concepts
    extractor = ConceptExtractor()
    paper.concepts = extractor.extract_from_paper(paper)
    
    # Save paper
    filepath = paper.save('research/papers')
    print(f"Paper added successfully: {filepath}")
    print(f"Extracted concepts: {', '.join(paper.concepts[:10])}")
    if len(paper.concepts) > 10:
        print(f"  ... and {len(paper.concepts) - 10} more")


def list_papers(args):
    """List all papers in the system."""
    papers_dir = Path(args.directory)
    
    if not papers_dir.exists():
        print(f"Directory not found: {papers_dir}")
        return
    
    paper_files = list(papers_dir.glob('*.json'))
    
    if not paper_files:
        print("No papers found.")
        return
    
    print(f"\nFound {len(paper_files)} papers:\n")
    
    for i, filepath in enumerate(paper_files, 1):
        paper = ResearchPaper.load(str(filepath))
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Year: {paper.year}")
        print(f"   Concepts: {len(paper.concepts)}")
        print()


def build_knowledge_graph(args):
    """Build a knowledge graph from all papers."""
    papers_dir = Path(args.papers_dir)
    
    if not papers_dir.exists():
        print(f"Directory not found: {papers_dir}")
        return
    
    paper_files = list(papers_dir.glob('*.json'))
    
    if not paper_files:
        print("No papers found.")
        return
    
    print(f"Building knowledge graph from {len(paper_files)} papers...")
    
    kg = KnowledgeGraph()
    extractor = ConceptExtractor()
    
    for filepath in paper_files:
        paper = ResearchPaper.load(str(filepath))
        paper_id = filepath.stem
        
        # Extract concepts if not already done
        if not paper.concepts:
            paper.concepts = extractor.extract_from_paper(paper)
            paper.save(args.papers_dir)
        
        # Add concepts to graph
        for concept in paper.concepts:
            kg.add_concept(concept)
            kg.link_paper_to_concept(concept, paper_id)
        
        # Add relationships between concepts
        relationships = extractor.identify_relationships(paper.concepts)
        for source, target, rel_type, weight in relationships:
            kg.add_relationship(source, target, rel_type, weight)
    
    # Save graph
    kg.save(args.output)
    
    print(f"\nKnowledge graph built successfully!")
    print(f"  Concepts: {len(kg)}")
    print(f"  Edges: {kg.graph.number_of_edges()}")
    print(f"  Saved to: {args.output}")
    
    # Show most central concepts
    central = kg.get_central_concepts(5)
    if central:
        print(f"\nMost central concepts:")
        for concept, score in central:
            print(f"  - {concept} (centrality: {score:.3f})")


def query_graph(args):
    """Query the knowledge graph for a concept."""
    if not Path(args.graph).exists():
        print(f"Knowledge graph not found: {args.graph}")
        print("Build it first using: ssi-codex build-graph")
        return
    
    kg = KnowledgeGraph.load(args.graph)
    
    concept = args.concept.lower()
    
    if concept not in kg.graph:
        print(f"Concept '{concept}' not found in knowledge graph.")
        
        # Suggest similar concepts
        similar = [c for c in kg.graph.nodes() if concept in c or c in concept]
        if similar:
            print(f"\nDid you mean one of these?")
            for c in similar[:5]:
                print(f"  - {c}")
        return
    
    print(f"\nConcept: {concept}")
    print(f"=" * 60)
    
    # Get related concepts
    related = kg.get_related_concepts(concept)
    if related:
        print(f"\nRelated concepts ({len(related)}):")
        for c in related[:10]:
            print(f"  - {c}")
        if len(related) > 10:
            print(f"  ... and {len(related) - 10} more")
    
    # Get associated papers
    papers = kg.get_papers_for_concept(concept)
    if papers:
        print(f"\nAssociated papers ({len(papers)}):")
        for paper_id in list(papers)[:5]:
            print(f"  - {paper_id}")
        if len(papers) > 5:
            print(f"  ... and {len(papers) - 5} more")


if __name__ == '__main__':
    sys.exit(main())
