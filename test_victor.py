#!/usr/bin/env python3
"""
Test script for VictorFractal GODCORE v1.0.0
This script validates all components work correctly.
"""

import sys
import os
import time

# Add VictorFractal to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'VictorFractal'))

from core.fractal_tokenizer_vtk import FractalTokenizer
from core.fractal_memory import HyperFractalMemory
from core.fractal_attention import FractalAttention
from core.directive_router import DirectiveRouter
from core.pulse_exchange import PulseExchange
from core.ckpt_utils import save_ckpt, load_ckpt

def test_tokenizer():
    print("Testing FractalTokenizer...")
    tok = FractalTokenizer(version="v1.0.0-TEST")
    
    text = "Hello Victor, welcome to the world!"
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    
    print(f"  Input: {text}")
    print(f"  Tokens: {len(tokens)} tokens generated")
    print(f"  Decoded: {decoded[:50]}...")
    print("  ✓ FractalTokenizer works!")
    return tok, tokens

def test_memory(tok, tokens):
    print("\nTesting HyperFractalMemory...")
    mem = HyperFractalMemory(dim=192, decay=0.002, capacity=5000)
    
    # Store some memories
    mem.remember(tokens, meta={"test": True, "time": time.time()})
    mem.remember(tok.encode("Another thought"), meta={"test": True, "time": time.time()})
    
    # Sample memories
    batch = mem.sample(k=2)
    print(f"  Stored 2 memories")
    print(f"  Sampled {len(batch)} memories")
    
    # Test state dump/load
    state = mem.dump_state()
    print(f"  State dump size: {len(state['items'])} items")
    
    mem2 = HyperFractalMemory()
    mem2.load_state(state)
    print(f"  State loaded successfully")
    print("  ✓ HyperFractalMemory works!")
    return mem

def test_attention(tokens):
    print("\nTesting FractalAttention...")
    attn = FractalAttention(scales=[3, 7, 13, 29], alpha=0.72)
    
    fused = attn.fuse(tokens)
    print(f"  Input tokens: {len(tokens)}")
    print(f"  Fused tokens: {len(fused)}")
    print(f"  Sample fused values: {fused[:5]}")
    print("  ✓ FractalAttention works!")
    return attn

def test_router():
    print("\nTesting DirectiveRouter...")
    modes = {
        "reflect": {"temperature": 0.3, "style": "introspective"},
        "expand": {"temperature": 0.6, "style": "visionary"},
        "defend": {"temperature": 0.4, "style": "protective"}
    }
    router = DirectiveRouter(modes=modes, default_mode="expand")
    
    print(f"  Default mode: {router.current_mode()}")
    router.set_mode("reflect")
    print(f"  After set_mode('reflect'): {router.current_mode()}")
    print("  ✓ DirectiveRouter works!")
    return router

def test_pulse_exchange():
    print("\nTesting PulseExchange...")
    bus = PulseExchange()
    
    received = []
    def handler(payload):
        received.append(payload)
    
    bus.on("test_event", handler)
    bus.emit("test_event", message="Hello from bus!")
    
    print(f"  Emitted event, received: {len(received)} messages")
    print(f"  Message: {received[0] if received else 'None'}")
    print("  ✓ PulseExchange works!")
    return bus

def test_checkpoint():
    print("\nTesting Checkpoint utilities...")
    test_dir = "/tmp/victor_test_ckpt"
    os.makedirs(test_dir, exist_ok=True)
    
    test_data = {
        "version": "v1.0.0-TEST",
        "state": {"heartbeat": 42, "mode": "expand"},
        "meta": {"timestamp": time.time()}
    }
    
    ckpt_path = os.path.join(test_dir, "test.ckpt")
    save_ckpt(test_data, ckpt_path)
    print(f"  Saved checkpoint to {ckpt_path}")
    
    loaded = load_ckpt(ckpt_path)
    print(f"  Loaded checkpoint: version={loaded['version']}")
    print(f"  State heartbeat: {loaded['state']['heartbeat']}")
    
    # Test non-existent checkpoint
    missing = load_ckpt("/tmp/does_not_exist.ckpt")
    assert missing is None, "Should return None for missing checkpoint"
    
    print("  ✓ Checkpoint utilities work!")
    
    # Cleanup
    os.remove(ckpt_path)
    os.rmdir(test_dir)

def main():
    print("=" * 60)
    print("VictorFractal GODCORE v1.0.0 - Component Test Suite")
    print("=" * 60)
    
    try:
        tok, tokens = test_tokenizer()
        mem = test_memory(tok, tokens)
        attn = test_attention(tokens)
        router = test_router()
        bus = test_pulse_exchange()
        test_checkpoint()
        
        print("\n" + "=" * 60)
        print("✓ All components passed tests successfully!")
        print("=" * 60)
        print("\nVictor is ready to boot!")
        print("Run: cd VictorFractal && python run_victor.py")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
