"""
ZPUE Interactive Visualization and Testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ssi_codex.zpue_enhanced import EnhancedZPUE
import json


def print_separator():
    print("\n" + "="*70 + "\n")


def demo_basic_zpue():
    """Demonstrate basic ZPUE functionality."""
    print("🧠 ZPUE Basic Demo - Learning from Scratch")
    print_separator()
    
    zpue = EnhancedZPUE(log_dir="logs/zpue_demo")
    
    # Training corpus
    corpus = [
        "artificial intelligence learns from data",
        "machine learning uses algorithms to learn patterns",
        "deep learning uses neural networks",
        "neural networks learn from examples",
        "data science extracts insights from data",
        "algorithms process data to learn",
        "intelligence emerges from learning"
    ]
    
    print("📚 Training ZPUE on corpus...")
    for i, sentence in enumerate(corpus, 1):
        print(f"  [{i}] Processing: '{sentence}'")
        zpue.update_model(sentence)
    
    print_separator()
    
    # Get statistics
    stats = zpue.get_stats()
    print("📊 ZPUE Statistics:")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Total tokens processed: {stats['total_tokens']}")
    print(f"  Unique bigrams: {stats['unique_bigrams']}")
    print(f"\n  Top 10 tokens:")
    for token, count in stats['top_tokens']:
        print(f"    {token}: {count}")
    
    print_separator()
    
    # Generate responses
    print("💬 Generating responses:")
    for _ in range(3):
        response, metadata = zpue.generate_response_with_semantics(length=8)
        print(f"  ZPUE: {response}")
    
    print_separator()
    
    # Semantic insights
    print("🔍 Semantic Insights:")
    test_tokens = ['learning', 'data', 'intelligence']
    for token in test_tokens:
        if token in zpue.vocab:
            insights = zpue.get_semantic_insights(token)
            print(f"\n  Token: '{token}'")
            print(f"    Frequency: {insights['frequency']}")
            print(f"    Similar tokens: {[t for t, _ in insights['similar_tokens'][:3]]}")
    
    print_separator()
    
    # Visualize semantic space
    print("📈 Semantic Space Visualization (PCA):")
    viz_data = zpue.visualize_semantic_space(method='pca', n_components=2)
    
    if 'error' not in viz_data:
        print(f"  Method: {viz_data['method']}")
        print(f"  Dimensions: {viz_data['n_components']}")
        print(f"  Tokens visualized: {len(viz_data['tokens'])}")
        print(f"  Explained variance: {[f'{v:.2%}' for v in viz_data['explained_variance']]}")
        
        # Show a few token coordinates
        print(f"\n  Sample coordinates:")
        for i in range(min(5, len(viz_data['tokens']))):
            token = viz_data['tokens'][i]
            coords = viz_data['coordinates'][i]
            print(f"    {token}: ({coords[0]:.3f}, {coords[1]:.3f})")
    else:
        print(f"  {viz_data['error']}")
    
    print_separator()
    
    # Interactive test
    print("🎯 Interactive Test:")
    test_inputs = [
        "what is learning",
        "data and intelligence",
        "neural networks are intelligent"
    ]
    
    for user_input in test_inputs:
        print(f"\n  User: {user_input}")
        result = zpue.interact_enhanced(user_input)
        print(f"  ZPUE: {result['response']}")
        print(f"  Vocab size: {result['stats']['vocab_size']}")
    
    print_separator()
    
    # Export state
    export_path = "logs/zpue_demo/zpue_state.json"
    zpue.export_state(export_path)
    print(f"💾 State exported to: {export_path}")
    
    print_separator()
    print("✅ Demo complete!")


def demo_research_integration():
    """Demonstrate ZPUE integration with research papers."""
    print("📚 ZPUE Research Integration Demo")
    print_separator()
    
    from ssi_codex import ResearchPaper
    
    zpue = EnhancedZPUE(log_dir="logs/zpue_research")
    
    # Load example paper
    try:
        paper = ResearchPaper.load("research/papers/2017_Attention_Is_All_You_Need.json")
        
        print(f"📄 Processing paper: {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}...")
        print()
        
        # Feed abstract to ZPUE
        print("🔄 Learning from abstract...")
        zpue.update_model(paper.abstract)
        
        # Feed keywords
        if paper.keywords:
            print("🔄 Learning from keywords...")
            zpue.update_model(" ".join(paper.keywords))
        
        stats = zpue.get_stats()
        print(f"\n📊 After processing paper:")
        print(f"   Vocabulary size: {stats['vocab_size']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        
        print_separator()
        
        # Generate summary-like responses
        print("📝 Generating learned responses:")
        for i in range(3):
            response, metadata = zpue.generate_response_with_semantics(length=12)
            print(f"  [{i+1}] {response}")
        
        print_separator()
        
        # Show what it learned about key concepts
        print("🧠 Learned concepts:")
        key_concepts = ['transformer', 'attention', 'model', 'networks']
        for concept in key_concepts:
            if concept in zpue.vocab:
                insights = zpue.get_semantic_insights(concept)
                print(f"\n  {concept}:")
                print(f"    Frequency: {insights['frequency']}")
                similar = insights['similar_tokens'][:3]
                if similar:
                    print(f"    Similar: {', '.join([t for t, _ in similar])}")
        
        print_separator()
        print("✅ Research integration demo complete!")
        
    except FileNotFoundError:
        print("⚠️  Example paper not found. Run from repository root.")
        print_separator()


def interactive_mode():
    """Interactive chat with ZPUE."""
    print("💬 ZPUE Interactive Mode")
    print("   Type 'quit' to exit")
    print("   Type 'stats' to see statistics")
    print("   Type 'insights <token>' to see semantic insights")
    print("   Type 'viz' to see semantic space")
    print_separator()
    
    zpue = EnhancedZPUE(log_dir="logs/zpue_interactive")
    
    print("ZPUE: Hello. Teach me by talking to me.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nZPUE: Goodbye! I have learned from our conversation.")
                break
            
            elif user_input.lower() == 'stats':
                stats = zpue.get_stats()
                print(f"\nZPUE Statistics:")
                print(f"  Vocabulary: {stats['vocab_size']} tokens")
                print(f"  Total processed: {stats['total_tokens']}")
                print(f"  Bigrams: {stats['unique_bigrams']}")
                print(f"  Top tokens: {', '.join([t for t, _ in stats['top_tokens'][:5]])}")
            
            elif user_input.lower().startswith('insights '):
                token = user_input[9:].strip()
                if token in zpue.vocab:
                    insights = zpue.get_semantic_insights(token)
                    print(f"\nInsights for '{token}':")
                    print(f"  Frequency: {insights['frequency']}")
                    print(f"  Similar tokens:")
                    for t, sim in insights['similar_tokens'][:5]:
                        print(f"    {t}: {sim:.3f}")
                else:
                    print(f"\nZPUE: I haven't learned the token '{token}' yet.")
            
            elif user_input.lower() == 'viz':
                viz = zpue.visualize_semantic_space()
                if 'error' not in viz:
                    print(f"\nSemantic space: {len(viz['tokens'])} tokens in {viz['n_components']}D")
                    print(f"Explained variance: {[f'{v:.1%}' for v in viz['explained_variance']]}")
                else:
                    print(f"\n{viz['error']}")
            
            else:
                result = zpue.interact_enhanced(user_input)
                print(f"\nZPUE: {result['response']}")
                
        except KeyboardInterrupt:
            print("\n\nZPUE: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point for ZPUE demos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZPUE Demonstrations")
    parser.add_argument('mode', choices=['basic', 'research', 'interactive', 'all'],
                       help='Demo mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'basic' or args.mode == 'all':
        demo_basic_zpue()
    
    if args.mode == 'research' or args.mode == 'all':
        demo_research_integration()
    
    if args.mode == 'interactive':
        interactive_mode()
    
    if args.mode == 'all':
        print("\n" + "🎉"*35)
        print("\n  All demos complete! Try interactive mode:")
        print("  python -m ssi_codex.zpue_demo interactive")
        print("\n" + "🎉"*35 + "\n")


if __name__ == "__main__":
    main()
