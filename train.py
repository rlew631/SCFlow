#!/usr/bin/env python3

"""
Refactored training script – pure PyTorch, no MMCV.

⚠️  Areas marked `TODO:` or `NOTE:` need a quick review because they reference
    local helpers (e.g. build_refiner, build_dataset) or features that MMCV
    handled for you (Runner logic, FP16 hooks, etc.).
"""

import importlib.util
from types import ModuleType
from typing import Any, Dict, Union, Callable

# ──────────────────────────────────────────────────────────────────────────────
# Std-lib
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import importlib.util
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (DataLoader, DistributedSampler, RandomSampler,
                              SequentialSampler, default_collate)

# ──────────────────────────────────────────────────────────────────────────────
# Local modules – these should already be *vanilla* imports in your repo
# ──────────────────────────────────────────────────────────────────────────────
from datasets import build_dataset       # NOTE: relies on local impl - first pass done
from models import build_refiner         # NOTE: relies on local impl - first pass done
from tools.eval import (                 # NOTE: relies on local impl - first pass done
    single_gpu_test,
    multi_gpu_test,
)

# If you actually use this, re-enable it and remove the pragma.
# from datasets import MultiSourceSampler  # noqa: F401  # TODO: supply vanilla version


# ──────────────────────────────────────────────────────────────────────────────
# Helpers that MMCV used to provide
# ──────────────────────────────────────────────────────────────────────────────
def _import_config_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)        # type: ignore[arg-type]
    return mod

def _is_public_value(v: Any) -> bool:
    """Filter out things we never want to stuff into cfg."""
    # skip callables (functions, classes), sub-modules, dunder / private vars
    from types import ModuleType  # local import to avoid polluting globals
    return not callable(v) and not isinstance(v, ModuleType)

def load_cfg(path: Union[str, "os.PathLike[str]"]) -> Dict[str, Any]:
    """
    Load a python config file.

    • If the file defines a top-level variable `cfg`, return it directly.
    • Otherwise, gather every **public** global (name not starting with '_')
      that is *not* a function/class/module into a dict and return that.

    This lets you delete the old `cfg = dict(...)` block from MMCV configs
    without breaking the new training script.
    """
    mod = _import_config_module(str(path))

    # Modern style: explicit cfg
    if hasattr(mod, "cfg"):
        return getattr(mod, "cfg")

    # Legacy style: pack loose globals
    cfg_dict: Dict[str, Any] = {
        name: value
        for name, value in vars(mod).items()
        if not name.startswith("_") and _is_public_value(value)
    }

    if not cfg_dict:
        raise RuntimeError(
            f"{path} contains no usable configuration variables. "
            "Either create a `cfg` dict or add public global settings."
        )
    return cfg_dict


def mkdir_or_exist(dir_path: str | os.PathLike) -> None:
    """Safely mimic mmcv.mkdir_or_exist."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def build_optimizer(model: torch.nn.Module, optim_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Minimal substitute for MMCV's build_optimizer."""
    optim_type = optim_cfg.pop("type")
    try:
        optim_cls = getattr(torch.optim, optim_type)
    except AttributeError as e:
        raise ValueError(f"Unknown optimizer type: {optim_type}") from e
    return optim_cls(model.parameters(), **optim_cfg)


# Very light wrapper; replace MMCV collate with default_collate unless custom
def mmcv_like_collate(batch, samples_per_gpu: int = 1):
    # TODO: If your code relies on mmcv.DataContainer, replace this.
    return default_collate(batch)


# Placeholder for EvalHook logic that MMCV used.
class EvalHook:
    def __init__(self, dataloader: DataLoader, test_fn, by_epoch: bool = True, **kwargs):
        self.dataloader = dataloader
        self.test_fn = test_fn
        self.by_epoch = by_epoch
        # kwargs may include eval frequency, etc.

    def after_train_epoch(self, runner):
        metrics = self.test_fn(runner.model, self.dataloader, device=runner.device)
        runner.logger.info(f"Validation metrics: {metrics}")


# Ultra-thin “Runner” so the rest of the file changes as little as possible.
class SimpleRunner:
    """MVP replacement for mmcv.runner.*Runner."""

    def __init__(self, model, optimizer, work_dir, logger, meta=None, **runner_cfg):
        self.model = model
        self.optimizer = optimizer
        self.work_dir = Path(work_dir)
        self.logger = logger
        self.max_epochs = runner_cfg.get("max_epochs", 1)
        self.device = next(model.parameters()).device
        self.hooks = []

    # Registration mirrors MMCV
    def register_hook(self, hook, priority="NORMAL"):
        self.hooks.append(hook)

    def register_training_hooks(
        self,
        lr_config,
        optimizer_config,
        checkpoint_config,
        log_config,
        momentum_config=None,
        custom_hooks_config=None,
    ):
        # TODO: Implement learning-rate schedulers, logging hooks, etc.
        pass

    def run(self, dataloaders, workflow):
        # Minimal train loop
        train_loader = dataloaders[0]
        for epoch in range(self.max_epochs):
            self.logger.info(f"Epoch [{epoch+1}/{self.max_epochs}]")
            self._train_one_epoch(train_loader, epoch)

            # Call hooks (e.g., validation)
            for hook in self.hooks:
                if hasattr(hook, "after_train_epoch"):
                    hook.after_train_epoch(self)

    def _train_one_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        for step, data in enumerate(dataloader):
            # TODO: adapt to your model’s expected input
            loss_dict = self.model.train_step(data)  # NOTE: depends on local API
            loss = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step % 10 == 0:  # crude logging
                self.logger.info(f"Epoch {epoch} - Iter {step} - loss: {loss.item():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI / main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train a pose refiner (Pure PyTorch)")
    parser.add_argument("--config", default="configs/refine_models/scflow.py", help="train config file path")
    parser.add_argument("--work-dir", type=str, help="working dir")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--launcher", default="none", choices=["none", "slurm", "mpi", "pytorch"], help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="if set, load pretrained model weights before training",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def build_dataloader(cfg, dataset, dataset_cfg, distributed: bool, shuffle: bool) -> DataLoader:
    """Re-implementation minus MMCV Samplers/Collate."""
    # NOTE: MultiSourceSampler not ported – implement if you used it.
    if distributed:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = torch.cuda.device_count()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        batch_size = cfg["data"]["samples_per_gpu"]
        num_workers = cfg["data"]["workers_per_gpu"]
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_size = cfg["data"]["samples_per_gpu"] * cfg["num_gpus"]
        num_workers = cfg["data"]["workers_per_gpu"] * cfg["num_gpus"]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(mmcv_like_collate, samples_per_gpu=cfg["data"]["samples_per_gpu"]),
        persistent_workers=False,
    )


def main() -> None:
    args = parse_args()

    # ------------------------- logging & directories -------------------------
    cfg = load_cfg(args.config)
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    mkdir_or_exist(cfg["work_dir"])

    log_file = Path(cfg["work_dir"]) / f'{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger("Flow-6D")
    logger.addHandler(logging.StreamHandler())  # also print to stdout
    logger.info("Starting script")

    # ------------------------- distributed setup ----------------------------
    distributed = args.launcher != "none"
    if distributed:
        # Vanilla torch DDP init
        torch.distributed.init_process_group(backend="nccl")  # adjust if needed
        world_size = torch.distributed.get_world_size()
        logger.info(f"Running distributed training on {world_size} GPUs")
    else:
        logger.info("Running single-machine training")

    # ------------------------- build model ----------------------------------
    logger.info("Building model …")
    model = build_refiner(cfg["model"])         # NOTE: must already be vanilla PyTorch

    if args.load_weights:
        logger.info("Initializing model weights …")
        model.init_weights()                    # NOTE: relies on your model's API
    else:
        logger.info("Not loading pretrained weights")

    # ------------------------- move to GPUs ---------------------------------
    if distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        model.cuda()
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False)
    else:
        gpu_ids = list(range(cfg["num_gpus"]))
        model = DataParallel(model.cuda(gpu_ids[0]), device_ids=gpu_ids)

    # ------------------------- optimizer & runner ---------------------------
    logger.info("Setting up optimizer")
    optimizer = build_optimizer(model, cfg["optimizer"])

    # fp16 support – left as TODO (requires Apex or torch.amp)
    if cfg.get("fp16"):                         # TODO: port fp16 logic if needed
        logger.warning("fp16 support not yet wired – skipping")

    runner = SimpleRunner(
        model=model,
        optimizer=optimizer,
        work_dir=cfg["work_dir"],
        logger=logger,
        **cfg["runner"],
    )

    # ------------------------- datasets & loaders ---------------------------
    train_dataset = build_dataset(cfg["data"]["train"])   # NOTE: must not rely on MMCV
    dataloaders = [build_dataloader(cfg, train_dataset, cfg["data"]["train"], distributed, shuffle=True)]

    # Validation
    if cfg.get("evaluation"):
        val_dataset = build_dataset(cfg["data"]["val"])
        val_loader = build_dataloader(cfg, val_dataset, cfg["data"]["val"], distributed, shuffle=False)
        eval_hook = EvalHook(val_loader, test_fn=multi_gpu_test if distributed else single_gpu_test)
        runner.register_hook(eval_hook, priority="LOW")

    # ------------------------- (Optional) resume ----------------------------
    if args.resume_from:
        logger.warning("--resume-from is provided but SimpleRunner.resume() isn't implemented")
        # TODO: custom checkpoint loading logic

    # ------------------------- run training ---------------------------------
    runner.run(dataloaders, cfg.get("work_flow", [("train", 1)]))


if __name__ == "__main__":
    main()
