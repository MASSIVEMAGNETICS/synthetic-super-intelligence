# ================================
# FILE: core/fractal_tokenizer_vtk.py
# VERSION: v1.1.0-SCOS-GODCORE
# NAME: FractalTokenizer
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Encode/decode text into FractalTokens (concept, intent, mood). Hash-based compact vocab.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ================================

import re
import hashlib
from typing import List

class FractalTokenizer:
    def __init__(self, version: str = "v1.1.0-SCOS-GODCORE", vocab_size: int = 8192):
        self.version = version
        self.vocab_size = vocab_size

    def _hash(self, s: str) -> int:
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % self.vocab_size

    def _pre(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-_'.,:;!?]", " ", text)
        toks = re.findall(r"[a-z0-9]+|[.,:;!?]", text)
        return toks

    def encode(self, text: str) -> List[int]:
        toks = self._pre(text)
        # include bigrams to inject local structure
        grams = []
        for i, t in enumerate(toks):
            grams.append(t)
            if i + 1 < len(toks):
                grams.append(t + "_" + toks[i+1])
        return [self._hash(g) for g in grams]

    def decode(self, ids: List[int]) -> str:
        # lossy reconstruction; for demo we just emit representative tokens
        return " ".join([f"τ{(i % 997)}" for i in ids])
