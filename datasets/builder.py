"""
dataset_registry.py
Minimal replacement for `mmcv.utils.Registry` and `build_from_cfg`.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Sequence, Type

from torch.utils.data import ConcatDataset, Dataset

# ────────────────────────────────────────────────────────────────
# 1. Lightweight Registry
# ────────────────────────────────────────────────────────────────
class Registry:
    """A very small re-implementation of MMCV's Registry."""
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    # --- registration helpers ------------------------------------------------
    def _register(self, cls: Type, module_name: Optional[str] = None) -> None:
        key = module_name or cls.__name__
        if key in self._module_dict:
            raise KeyError(f"{key} is already registered in {self._name}")
        self._module_dict[key] = cls

    def register_module(self, name: Optional[str] = None):
        """Decorator usage:

        @DATASETS.register_module()
        class MyDataset(Dataset): ...
        """
        def _decorator(cls: Type):
            self._register(cls, module_name=name)
            return cls
        return _decorator

    # --- getters -------------------------------------------------------------
    def get(self, key: str) -> Type:
        if key not in self._module_dict:
            raise KeyError(f"{key} is not registered in {self._name}")
        return self._module_dict[key]

    def __contains__(self, key: str) -> bool:  # optional convenience
        return key in self._module_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, items={list(self._module_dict)})"


# ────────────────────────────────────────────────────────────────
# 2. Vanilla `build_from_cfg`
# ────────────────────────────────────────────────────────────────
def build_from_cfg(cfg: Dict[str, Any], registry: Registry, **default_kwargs) -> Any:
    """Instantiate an object from a config dict.

    Expected format
    ---------------
    cfg = dict(
        type='ClassNameOrAlias',
        arg1=value1,
        arg2=value2,
        ...
    )
    """
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise TypeError("Config must be a dict with the key 'type'")

    cfg = cfg.copy()                 # avoid side-effects
    obj_type = cfg.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"`type` must be a str or class, not {type(obj_type)}")

    return obj_cls(**{**default_kwargs, **cfg})


# ────────────────────────────────────────────────────────────────
# 3. Registries identical to the MMCV ones you had
# ────────────────────────────────────────────────────────────────
DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")          # ← defined for parity; not used below


# ────────────────────────────────────────────────────────────────
# 4. Public helper
# ────────────────────────────────────────────────────────────────
def build_dataset(cfg: Dict[str, Any] | Sequence[Dict[str, Any]]) -> Dataset:
    """Recursively build dataset(s) from config.

    Two supported forms
    -------------------
    1. Single dataset:
       cfg = { 'type': 'MyDataset', 'root': '...', ... }

    2. Concatenation:
       cfg = {
           'type': 'ConcatDataset',
           'datasets': [cfg1, cfg2, ...]
       }
    """
    if isinstance(cfg, (list, tuple)):
        # Allow passing a bare list and implicitly wrap with ConcatDataset
        return ConcatDataset([build_dataset(c) for c in cfg])

    if cfg.get("type") == "ConcatDataset":
        inner = [build_dataset(c) for c in cfg["datasets"]]
        # ConcatDataset itself doesn’t need any extra args
        return ConcatDataset(inner)

    # Otherwise treat as a normal single dataset
    return build_from_cfg(cfg, DATASETS)


# ────────────────────────────────────────────────────────────────
# 5. Usage example (remove in production)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example registration
    @DATASETS.register_module()
    class Dummy(Dataset):
        def __init__(self, size: int = 10):
            self.size = size
        def __len__(self): return self.size
        def __getitem__(self, idx): return idx

    cfg_single = dict(type="Dummy", size=5)
    cfg_concat = dict(type="ConcatDataset", datasets=[cfg_single, cfg_single])

    ds1 = build_dataset(cfg_single)
    ds2 = build_dataset(cfg_concat)
    print(ds1, len(ds1))             # -> <Dummy ...> 5
    print(ds2, len(ds2))             # -> <torch.utils.data.ConcatDataset ...> 10
