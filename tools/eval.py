# utils/test_utils.py
"""
Utilities for single- and multi-GPU testing (MMCV-free version).

External deps: PyTorch, NumPy, tqdm (for nice progress-bars).
"""

from __future__ import annotations

import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
#                              Tiny MMCV Replacements
# -----------------------------------------------------------------------------
def get_dist_info() -> tuple[int, int]:
    """Return (rank, world_size) – works even when DDP isn't initialised."""
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


class ProgressBar:
    """Very small wrapper around tqdm so the old `mmcv.ProgressBar` calls stay."""
    def __init__(self, total: int):
        self._pbar = tqdm(total=total, ncols=80)

    def update(self, n: int = 1):
        self._pbar.update(n)

    def close(self):
        self._pbar.close()

    # ensures bar closes on garbage-collection
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def mkdir_or_exist(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def dump(obj: Any, file: str | os.PathLike) -> None:
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load(file: str | os.PathLike) -> Any:
    with open(file, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
#                               Helper functions
# -----------------------------------------------------------------------------
def format_result(batch_preds: Dict[str, torch.Tensor]) -> List[Dict[str, np.ndarray]]:
    """Convert batched tensor predictions to list-dict (per sample) of numpy."""
    results_batch: list[dict[str, np.ndarray]] = []
    random_key = random.choice(list(batch_preds.keys()))
    batch_size = len(batch_preds[random_key])

    for i in range(batch_size):
        sample_dict = {k: v[i].detach().cpu().numpy() for k, v in batch_preds.items()}
        results_batch.append(sample_dict)
    return results_batch


# -----------------------------------------------------------------------------
#                               Main test loops
# -----------------------------------------------------------------------------
@torch.no_grad()
def single_gpu_test(
    model: torch.nn.Module,
    data_loader: DataLoader,
    validate: bool = True,
) -> list[dict]:
    """Run inference on a single GPU (or CPU) and collect results locally."""
    model.eval()
    results: list[dict] = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))

    for data in data_loader:
        batch_preds = model(data, return_loss=False)
        formatted = format_result(batch_preds)
        batch_size = len(formatted)

        img_metas = data["img_metas"].data[0]
        if validate:
            for i in range(batch_size):
                results.append(dict(pred=formatted[i], gt={}, img_metas=img_metas[i]))
        else:
            for i in range(batch_size):
                results.append(dict(pred=formatted[i], img_metas=img_metas[i]))

        prog_bar.update(batch_size)
    prog_bar.close()
    return results


@torch.no_grad()
def multi_gpu_test(
    model: torch.nn.Module,
    data_loader: DataLoader,
    validate: bool = True,
    tmpdir: str | os.PathLike | None = None,
    gpu_collect: bool = False,
) -> list[dict] | None:
    """Distributed inference. Results are gathered to rank-0."""
    model.eval()
    results: list[dict] = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    prog_bar = ProgressBar(len(dataset)) if rank == 0 else None

    for data in data_loader:
        batch_preds = model(data, return_loss=False)
        formatted = format_result(batch_preds)
        batch_size = len(formatted)
        img_metas = data["img_metas"].data[0]

        if validate:
            for i in range(batch_size):
                results.append(dict(pred=formatted[i], gt={}, img_metas=img_metas[i]))
        else:
            for i in range(batch_size):
                results.append(dict(pred=formatted[i], img_metas=img_metas[i]))

        if prog_bar:
            prog_bar.update(batch_size * world_size)

    if prog_bar:
        prog_bar.close()

    # gather results across ranks
    if gpu_collect:
        return collect_results_gpu(results, len(dataset))
    return collect_results_cpu(results, len(dataset), tmpdir)


# -----------------------------------------------------------------------------
#               CPU / GPU result collection (rank-0 only returns)
# -----------------------------------------------------------------------------
def collect_results_cpu(
    result_part: list[dict],
    size: int,
    tmpdir: str | os.PathLike | None = None,
) -> list[dict] | None:
    rank, world_size = get_dist_info()

    # --- prepare tmp dir (rank-0 creates) ------------------------------------
    if tmpdir is None:
        if rank == 0:
            mkdir_or_exist(".dist_test")
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
        tmpdir_tensor = torch.tensor(
            bytearray(str(tmpdir).encode() if rank == 0 else b""),
            dtype=torch.uint8,
            device="cuda",
        )
        # broadcast path
        dist.broadcast(tmpdir_tensor, 0)
        tmpdir = tmpdir_tensor.cpu().numpy().tobytes().decode().rstrip("\x00")
    else:
        if rank == 0:
            mkdir_or_exist(tmpdir)

    # --- dump local results --------------------------------------------------
    dump(result_part, Path(tmpdir) / f"part_{rank}.pkl")
    dist.barrier()

    # --- rank-0 gathers ------------------------------------------------------
    if rank != 0:
        return None

    part_list = [load(Path(tmpdir) / f"part_{i}.pkl") for i in range(world_size)]
    ordered: list[dict] = [item for chunk in zip(*part_list) for item in chunk]
    shutil.rmtree(tmpdir)
    return ordered[:size]


def collect_results_gpu(result_part: list[dict], size: int) -> list[dict] | None:
    rank, world_size = get_dist_info()

    # serialise with pickle → tensor of uint8
    part_tensor = torch.tensor(bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda")

    # share shapes
    shape_tensor = torch.tensor([part_tensor.numel()], device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    max_len = int(torch.tensor(shape_list).max())
    padded = torch.zeros(max_len, dtype=torch.uint8, device="cuda")
    padded[: part_tensor.numel()] = part_tensor

    part_recv = [torch.empty(max_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
    dist.all_gather(part_recv, padded)

    if rank != 0:
        return None

    part_list = [
        pickle.loads(buf[: shape.item()].cpu().numpy().tobytes())
        for buf, shape in zip(part_recv, shape_list)
    ]
    ordered: list[dict] = [item for chunk in zip(*part_list) for item in chunk]
    return ordered[:size]


# -----------------------------------------------------------------------------
#               Miscellaneous (IoU / area) utilities – unchanged
# -----------------------------------------------------------------------------
def intersect_and_union(pred_mask: np.ndarray | torch.Tensor,
                        gt_mask: np.ndarray | torch.Tensor):
    """Compute intersection / union areas for binary masks (broadcast-aware)."""
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    # shapes: (N, H, W) and (M, H, W)  →  broadcast to (N, M, H, W)
    intersect = pred_mask[:, None] & gt_mask[None]
    area_intersect = intersect.sum(axis=(-1, -2))             # (N, M)

    area_pred = pred_mask.sum(axis=(-1, -2))                  # (N,)
    area_gt = gt_mask.sum(axis=(-1, -2))                      # (M,)
    area_union = area_pred[:, None] + area_gt[None] - area_intersect

    return area_intersect, area_union, area_pred, area_gt
