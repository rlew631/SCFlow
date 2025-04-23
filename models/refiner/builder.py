"""
refiner_registry.py
Lightweight substitute for MMCV's Registry + build_from_cfg, focused on refiners.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

# ────────────────────────────────────────────────────────────────
# 1. Minimal Registry (same code as before; keep only once in project)
# ────────────────────────────────────────────────────────────────
class Registry:
    def __init__(self, name: str):
        self._name = name
        self._modules: dict[str, Type] = {}

    # decorator-style registration
    def register_module(self, name: Optional[str] = None):
        def _decorator(cls: Type):
            key = name or cls.__name__
            if key in self._modules:
                raise KeyError(f"{key} already in {self._name}")
            self._modules[key] = cls
            return cls
        return _decorator

    # fetch
    def get(self, key: str) -> Type:
        if key not in self._modules:
            raise KeyError(f"{key} not found in {self._name}")
        return self._modules[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name}, items={list(self._modules)})"


# ────────────────────────────────────────────────────────────────
# 2. build_from_cfg helper (identical to earlier version)
# ────────────────────────────────────────────────────────────────
def build_from_cfg(cfg: Dict[str, Any], registry: Registry) -> Any:
    cfg = cfg.copy()
    obj_type = cfg.pop("type")
    obj_cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
    return obj_cls(**cfg)


# ────────────────────────────────────────────────────────────────
# 3. Public registry + factory
# ────────────────────────────────────────────────────────────────
REFINERS = Registry("refiner")


def build_refiner(cfg: Dict[str, Any]):
    """Instantiate a refiner based on a config dict.

    Example
    -------
    cfg = dict(
        type='MyRefiner',
        hidden_dim=256,
        num_layers=6,
    )
    """
    return build_from_cfg(cfg, REFINERS)


# ────────────────────────────────────────────────────────────────
# 4. Example usage (remove in production)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Register a dummy refiner
    @REFINERS.register_module()
    class ToyRefiner:
        def __init__(self, hidden_dim: int = 128):
            self.hidden_dim = hidden_dim
        def __repr__(self) -> str:
            return f"ToyRefiner(hidden_dim={self.hidden_dim})"

    cfg = dict(type="ToyRefiner", hidden_dim=512)
    refiner = build_refiner(cfg)
    print(refiner)         # -> ToyRefiner(hidden_dim=512)
