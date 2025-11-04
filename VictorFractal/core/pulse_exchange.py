# ================================
# FILE: core/pulse_exchange.py
# VERSION: v1.0.0-GODCORE
# NAME: PulseExchange
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Minimal in-process event bus (future: multi-agent signaling).
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ================================

from collections import defaultdict
from typing import Callable, Dict, List

class PulseExchange:
    def __init__(self):
        self.subs: Dict[str, List[Callable]] = defaultdict(list)

    def on(self, topic: str, fn: Callable):
        self.subs[topic].append(fn)

    def emit(self, topic: str, **payload):
        for fn in self.subs.get(topic, []):
            fn(payload)
