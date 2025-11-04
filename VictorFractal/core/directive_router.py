# ================================
# FILE: core/directive_router.py
# VERSION: v1.0.0-GODCORE
# NAME: DirectiveRouter
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Mode switching for cognition styles (reflect/expand/defend).
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ================================

class DirectiveRouter:
    def __init__(self, modes: dict, default_mode: str = "expand"):
        self.modes = modes
        self._mode = default_mode

    def set_mode(self, name: str):
        if name in self.modes:
            self._mode = name

    def current_mode(self) -> str:
        return self._mode
