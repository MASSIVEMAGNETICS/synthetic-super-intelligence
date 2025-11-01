#!/usr/bin/env python3
"""
Demonstration script for the SSI Codex Bando Fractal Model integration.
This shows how the self-evolving AGI system integrates with the research framework.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_basic_integration():
    """Demonstrate basic SSI Codex functionality."""
    print("="*70)
    print("SSI CODEX - Synthetic Super Intelligence Research System")
    print("="*70)
    print()
    
    from ssi_codex import ResearchPaper, KnowledgeGraph, ConceptExtractor
    
    # Create a research paper
    print("1. Creating a research paper...")
    paper = ResearchPaper(
        title="Recursive Self-Improvement in AGI Systems",
        authors=["Brandon Emery", "Victor"],
        abstract="This paper explores recursive self-improvement mechanisms in artificial general intelligence systems, including genetic algorithms, neural evolution, and meta-learning approaches.",
        year=2025,
        keywords=["AGI", "self-improvement", "meta-learning", "genetic algorithms"]
    )
    
    # Extract concepts
    print("2. Extracting concepts...")
    extractor = ConceptExtractor()
    paper.concepts = extractor.extract_from_paper(paper)
    print(f"   Extracted {len(paper.concepts)} concepts")
    print(f"   Top concepts: {paper.concepts[:5]}")
    
    # Save paper
    print("3. Saving paper...")
    filepath = paper.save("research/papers")
    print(f"   Saved to: {filepath}")
    
    # Build knowledge graph
    print("4. Building knowledge graph...")
    kg = KnowledgeGraph()
    for concept in paper.concepts:
        kg.add_concept(concept)
        kg.link_paper_to_concept(concept, paper.title)
    
    print(f"   Knowledge graph has {len(kg)} concepts")
    
    # Find central concepts
    central = kg.get_central_concepts(top_n=5)
    if central:
        print("   Central concepts:")
        for concept, score in central:
            print(f"     - {concept}: {score:.3f}")
    
    print()

def demo_fractal_model():
    """Demonstrate the Bando Fractal Model if available."""
    print("="*70)
    print("BANDO SUPER FRACTAL LANGUAGE MODEL - Self-Evolving AGI")
    print("="*70)
    print()
    
    try:
        from ssi_codex import BandoSuperFractalLanguageModel
        
        print("✓ BandoSuperFractalLanguageModel available")
        print()
        
        # Initialize model
        print("1. Initializing SFLM...")
        sflm = BandoSuperFractalLanguageModel()
        print()
        
        # Test basic processing
        print("2. Processing research query...")
        result = sflm.step(
            "What are the key challenges in artificial general intelligence?",
            mode="evolve"
        )
        
        print(f"   Intent: {result['token_info']['intent']}")
        print(f"   Tokens processed: {result['token_info']['length']}")
        print(f"   Kernel output: {result['kernel_output']}")
        print(f"   Duration: {result['duration_sec']:.4f}s")
        print()
        
        # Test self-modification detection
        print("3. Testing self-modification detection...")
        result = sflm.step(
            "self.mutate the optimization algorithm to improve convergence",
            mode="evolve"
        )
        
        print(f"   Intent: {result['token_info']['intent']}")
        print(f"   Self-modifying: {'Yes' if 'birth' in result['memory_event_id'] else 'No'}")
        print(f"   Output: {result['transformer_mesh_output_summary'][:100]}...")
        print()
        
        # Show growth stats
        print(f"4. Growth Engine Status:")
        print(f"   Generation: {sflm.growth_engine.generation}")
        print(f"   Best fitness: {sflm.growth_engine.best_fitness}")
        print(f"   Code variants: {len(sflm.growth_engine.code_reps)}")
        print()
        
        # Show run history
        print("5. Run History:")
        summary = sflm.summary(n=2)
        for i, run in enumerate(summary, 1):
            print(f"   Run {i}:")
            print(f"     Input: {run['input'][:60]}...")
            print(f"     Duration: {run['duration_sec']:.4f}s")
            print(f"     Events: {len(sflm.memory.events)}")
        
        return True
        
    except ImportError as e:
        print(f"✗ BandoSuperFractalLanguageModel not available")
        print(f"  Reason: {e}")
        print(f"  Install requirements: pip install numpy torch")
        return False

def demo_integration():
    """Demonstrate integration between components."""
    print()
    print("="*70)
    print("INTEGRATED WORKFLOW - Fractal Model + Research System")
    print("="*70)
    print()
    
    try:
        from ssi_codex import (
            BandoSuperFractalLanguageModel,
            ResearchPaper,
            ConceptExtractor,
            KnowledgeGraph
        )
        
        # Load existing paper
        print("1. Loading research paper...")
        paper = ResearchPaper.load("research/papers/2017_Attention_Is_All_You_Need.json")
        print(f"   Paper: {paper.title}")
        print(f"   Year: {paper.year}")
        print()
        
        # Process with fractal model
        print("2. Processing with SFLM...")
        sflm = BandoSuperFractalLanguageModel()
        
        input_text = f"Analyze this research: {paper.title}. {paper.abstract[:200]}"
        result = sflm.step(input_text, context={"paper_id": paper.title})
        
        print(f"   Processed {result['token_info']['length']} tokens")
        print(f"   Intent: {result['token_info']['intent']}")
        print()
        
        # Extract concepts from SFLM output
        print("3. Extracting concepts from SFLM analysis...")
        extractor = ConceptExtractor()
        sflm_concepts = extractor.extract_from_text(
            result['transformer_mesh_output_summary']
        )
        print(f"   Extracted {len(sflm_concepts)} concepts from SFLM output")
        
        # Combine with paper concepts
        all_concepts = set(paper.concepts + sflm_concepts)
        print(f"   Total unique concepts: {len(all_concepts)}")
        print()
        
        # Build enhanced knowledge graph
        print("4. Building enhanced knowledge graph...")
        kg = KnowledgeGraph()
        for concept in all_concepts:
            kg.add_concept(concept)
            kg.link_paper_to_concept(concept, paper.title)
        
        print(f"   Knowledge graph: {len(kg)} concepts")
        
        central = kg.get_central_concepts(top_n=3)
        if central:
            print("   Most central concepts:")
            for concept, score in central:
                print(f"     - {concept}: {score:.3f}")
        
        print()
        print("✓ Integration successful!")
        
    except ImportError:
        print("  Skipping integration demo (SFLM not available)")

if __name__ == "__main__":
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  SSI CODEX - Comprehensive System Demonstration".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    # Run demos
    demo_basic_integration()
    fractal_available = demo_fractal_model()
    
    if fractal_available:
        demo_integration()
    
    print()
    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print()
    print("Next steps:")
    print("  - Add more research papers with: ssi-codex add")
    print("  - Build knowledge graph with: ssi-codex build-graph")
    print("  - Query concepts with: ssi-codex query --concept 'your-concept'")
    if fractal_available:
        print("  - Explore evolved code in: sanctum/")
        print("  - Review growth log: victor_growth.log")
    print()
