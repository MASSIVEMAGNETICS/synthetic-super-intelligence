# ================================
# FILE: core/ckpt_utils.py
# VERSION: v1.0.0-GODCORE
# NAME: CheckpointUtils
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Save/Load lightweight .ckpt (pickle-based) for Victor state.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ================================

import os
import pickle

def save_ckpt(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_ckpt(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
