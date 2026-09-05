#!/usr/bin/env python3
"""Decide whether workshop train jobs may run concurrently.

This machine's Prompt-0 hardware is the source of truth: do not oversubscribe
a single accelerator. Prints JSON to stdout and writes outputs/hw_train_plan.json.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mem_bytes() -> int:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    except Exception:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page) * int(pages)


def _plan() -> dict:
    import torch

    cuda_n = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    mem = _mem_bytes()
    mem_gb = mem / (1024**3)
    n_accel = cuda_n if cuda_n else (1 if mps else 0)
    force43 = os.environ.get("FINISH_PARALLEL_SEED43", "") == "1"
    force42 = os.environ.get("FINISH_PARALLEL_SEED42_MSE", "") == "1"
    force_cpu = os.environ.get("FORCE_CPU_PARALLEL", "") == "1"

    # Two workshop jobs each pin ~1.6 GB of float32 frames plus activations.
    # Unified-memory Apple boxes with ≤16 GB will swap if CPU trains beside MPS.
    ram_ok_for_cpu_sidecar = mem_gb >= 24 and force_cpu

    parallel_seed42 = False
    parallel_seed43 = False
    seed42_mse_device = "auto"
    seed43_vic_device = "auto"
    seed43_mse_device = "auto"
    reasons: list[str] = []

    if cuda_n >= 2:
        parallel_seed42 = True
        parallel_seed43 = True
        seed42_mse_device = "cuda:1"
        seed43_vic_device = "cuda:0"
        seed43_mse_device = "cuda:1"
        reasons.append(f"{cuda_n} CUDA devices: pair jobs on cuda:0 and cuda:1.")
    elif cuda_n == 1:
        reasons.append("One CUDA device. Training jobs stay sequential on cuda:0.")
        seed42_mse_device = "cuda:0"
        seed43_vic_device = "cuda:0"
        seed43_mse_device = "cuda:0"
    elif mps:
        reasons.append(
            "One Metal/MPS device (no CUDA). Two workshop ViTs on the same GPU "
            f"would oversubscribe; RAM={mem_gb:.1f} GiB unified."
        )
        if ram_ok_for_cpu_sidecar:
            parallel_seed42 = True
            parallel_seed43 = True
            seed42_mse_device = "cpu"
            seed43_vic_device = "mps"
            seed43_mse_device = "cpu"
            reasons.append("FORCE_CPU_PARALLEL=1 and ≥24 GiB RAM: CPU sidecar allowed.")
        else:
            reasons.append(
                "CPU sidecar refused: 200-epoch ViT on CPU is far slower than MPS "
                "and would contend for unified memory with the live MPS job."
            )
        seed42_mse_device = seed42_mse_device if parallel_seed42 else "mps"
        seed43_vic_device = "mps"
        seed43_mse_device = seed43_mse_device if parallel_seed43 else "mps"
    else:
        reasons.append("CPU only. Jobs stay sequential (one train process).")
        seed42_mse_device = "cpu"
        seed43_vic_device = "cpu"
        seed43_mse_device = "cpu"

    if force42 and not parallel_seed42:
        reasons.append("FINISH_PARALLEL_SEED42_MSE=1 ignored: hardware cannot pair jobs.")
    if force43 and not parallel_seed43:
        reasons.append("FINISH_PARALLEL_SEED43=1 ignored: hardware cannot pair jobs.")

    skip43 = os.environ.get("SKIP_SEED43", "") == "1"
    skip_pixel = os.environ.get("SKIP_PIXEL_AE", "") == "1"

    return {
        "n_accelerators": n_accel,
        "cuda_count": cuda_n,
        "mps": mps,
        "ram_gib": round(mem_gb, 2),
        "parallel_seed42_mse_with_vicreg": parallel_seed42,
        "parallel_seed43_vicreg_mse": parallel_seed43,
        "seed42_mse_device": seed42_mse_device,
        "seed43_vic_device": seed43_vic_device,
        "seed43_mse_device": seed43_mse_device,
        "skip_seed43": skip43,
        "skip_pixel_ae": skip_pixel,
        "reason": " ".join(reasons),
    }


def main() -> dict:
    plan = _plan()
    dest = ROOT / "outputs" / "hw_train_plan.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return plan


if __name__ == "__main__":
    main()
