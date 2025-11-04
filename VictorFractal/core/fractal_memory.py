# ================================
# FILE: core/fractal_memory.py
# VERSION: v1.0.0-GODCORE
# NAME: HyperFractalMemory
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Lightweight vector memory with time decay + rolling capacity.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ================================

import time
import numpy as np
from typing import Dict, Any, List

class HyperFractalMemory:
    def __init__(self, dim: int = 192, decay: float = 0.002, capacity: int = 5000):
        self.dim = dim
        self.decay = decay
        self.capacity = capacity
        self.store: List[Dict[str, Any]] = []
        # Random projection seed
        rng = np.random.default_rng(37)
        self.R = rng.normal(size=(8192, dim)).astype(np.float32)

    def _embed(self, token_ids: List[int]) -> np.ndarray:
        # sum of random projections → simple semantic sketch
        vec = np.zeros(self.dim, dtype=np.float32)
        for tid in token_ids[:512]:
            vec += self.R[tid % self.R.shape[0]]
        # L2 normalize
        n = np.linalg.norm(vec) + 1e-8
        return vec / n

    def remember(self, tokens: List[int], meta: Dict[str, Any]):
        now = time.time()
        vec = self._embed(tokens)
        self.store.append({"t": now, "v": vec, "tokens": tokens, "meta": meta})
        # capacity control
        if len(self.store) > self.capacity:
            self.store = self.store[-self.capacity:]

    def sample(self, k: int = 5) -> List[Dict[str, Any]]:
        # time-decayed sampling weights
        if not self.store:
            return []
        now = time.time()
        weights = []
        for item in self.store:
            age = now - item["t"]
            w = np.exp(-self.decay * age)
            weights.append(w)
        w = np.array(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)
        idx = np.random.choice(len(self.store), size=min(k, len(self.store)), replace=False, p=w)
        return [self.store[i] for i in idx]

    def dump_state(self) -> Dict[str, Any]:
        # compact export: only vectors + minimal meta
        return {
            "dim": self.dim,
            "decay": self.decay,
            "capacity": self.capacity,
            "items": [
                {"t": it["t"], "v": it["v"].tolist(), "meta": it["meta"]} for it in self.store[-1000:]
            ]
        }

    def load_state(self, state: Dict[str, Any]):
        self.dim = state.get("dim", self.dim)
        self.decay = state.get("decay", self.decay)
        self.capacity = state.get("capacity", self.capacity)
        self.store = []
        for rec in state.get("items", []):
            self.store.append({"t": rec["t"], "v": np.array(rec["v"], dtype=np.float32), "tokens": [], "meta": rec.get("meta", {})})
