# =================================================================================================
# FILE: zpue.py
# VERSION: v0.1.0-GODCORE
# NAME: ZeroPointUnderstandingEngine
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Zero-Point Understanding Engine that learns natural language from scratch
# LICENSE: Proprietary – Massive Magnetics / Ethica AI / BHeard Network
# INTEGRATED: SSI Codex Research System
# =================================================================================================

from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Logging Configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [ZPUE] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class NGramModel:
    """Simple n-gram language model that updates counts online.

    Maintains counts of uni-grams and bi-grams observed in the corpus.
    Provides a method to predict the next token given the previous token
    by sampling from observed frequencies.
    """

    unigram_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bigram_counts: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    total_tokens: int = 0

    def update(self, tokens: List[str]) -> None:
        """Update n-gram counts with a list of tokens."""
        prev_token = None
        for token in tokens:
            self.unigram_counts[token] += 1
            self.total_tokens += 1
            if prev_token is not None:
                self.bigram_counts[(prev_token, token)] += 1
            prev_token = token

    def predict_next(self, prev_token: Optional[str]) -> str:
        """Predict the next token given the previous token.

        If a bigram exists for the previous token, sample from the bigram
        distribution; otherwise, sample from the unigram distribution.
        If the model has not learned anything yet, returns a placeholder.
        """
        if self.total_tokens == 0:
            return "..."
        if prev_token and any(k[0] == prev_token for k in self.bigram_counts):
            candidates = [(tok, count) for (prev, tok), count in self.bigram_counts.items() if prev == prev_token]
        else:
            candidates = list(self.unigram_counts.items())
        
        if not candidates:
            return "..."
            
        tokens, counts = zip(*candidates)
        total = sum(counts)
        probs = [c / total for c in counts]
        return random.choices(tokens, probs)[0]


class ZeroPointUnderstandingEngine:
    """Engine for incremental natural language understanding and generation.

    This class ties together a dynamic vocabulary, an n-gram model, and
    functions to process user input and generate responses. The engine
    tokenizes input text using a simple regex tokenizer, updates its model
    online, and generates a response by sampling from its learned token
    distribution. Over time, the engine builds a richer vocabulary and
    grammar based solely on interaction with the user.
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self.vocab: Dict[str, int] = {}
        self.model = NGramModel()
        self.prev_token: Optional[str] = None
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ZPUE initialized")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize a string into lowercased words using a simple regex."""
        return re.findall(r"\b\w+\b", text.lower())

    def update_model(self, text: str) -> None:
        """Update the vocabulary and n-gram model with new text."""
        tokens = self.tokenize(text)
        for token in tokens:
            self.vocab[token] = self.vocab.get(token, 0) + 1
        self.model.update(tokens)
        logger.debug(f"Updated model with tokens: {tokens}")

    def generate_response(self, length: int = 10) -> str:
        """Generate a response by sampling tokens from the n-gram model."""
        if self.model.total_tokens == 0:
            return "I have nothing to say yet."
        tokens: List[str] = []
        prev = self.prev_token
        for _ in range(length):
            token = self.model.predict_next(prev)
            tokens.append(token)
            prev = token
        response = " ".join(tokens)
        # Update prev_token for next interaction
        self.prev_token = tokens[-1] if tokens else None
        return response

    def interact(self, input_text: str) -> str:
        """Process user input, update model, and return a generated response."""
        self.update_model(input_text)
        response = self.generate_response()
        if self.log_dir:
            self._log_interaction(input_text, response)
        return response

    def _log_interaction(self, user_input: str, response: str) -> None:
        """Log the interaction to a JSON file."""
        if not self.log_dir:
            return
        file_path = self.log_dir / f"zpue_{len(self.model.unigram_counts):05d}.json"
        with open(file_path, "w") as f:
            json.dump({"input": user_input, "response": response}, f, indent=2)
        logger.info(f"Saved interaction to {file_path}")

    def get_stats(self) -> Dict:
        """Get statistics about the model's current state."""
        return {
            "vocab_size": len(self.vocab),
            "total_tokens": self.model.total_tokens,
            "unique_bigrams": len(self.model.bigram_counts),
            "top_tokens": sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)[:10]
        }


def main() -> None:
    """Command-line interface for interacting with the ZPUE engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a Zero-Point Understanding Engine interactive session.")
    parser.add_argument("--log", action="store_true", help="Enable logging of interactions to ./logs/zpue.")
    args = parser.parse_args()

    log_dir = None
    if args.log:
        log_dir = os.path.join("logs", "zpue")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    engine = ZeroPointUnderstandingEngine(log_dir=log_dir)
    print("ZPUE: Hello. Teach me.")
    try:
        while True:
            user_input = input("You: ")
            if not user_input:
                continue
            response = engine.interact(user_input)
            print("ZPUE:", response)
    except KeyboardInterrupt:
        print("\nZPUE: Goodbye.")


if __name__ == "__main__":
    main()
