#!/usr/bin/env python3
"""Verify the actual CUDA Mamba forward/backward kernel and save its provenance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mamba_jaad_utils import RESULTS_ROOT, environment_report, write_json
from scripts.trajectory_preserving_utils import sha256_file


def main():
    import mamba_ssm
    import selective_scan_cuda
    from mamba_ssm import Mamba

    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA smoke test is required")
    x = torch.randn(4, 15, 128, device="cuda", requires_grad=True)
    model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).cuda()
    y = model(x)
    y.square().mean().backward()
    report = environment_report()
    report.update({
        "mamba_import_version": mamba_ssm.__version__, "input_shape": list(x.shape), "output_shape": list(y.shape),
        "output_finite": bool(torch.isfinite(y).all()), "input_gradient_finite": bool(torch.isfinite(x.grad).all()),
        "parameter_gradients_finite": all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
        "selective_scan_binary": str(selective_scan_cuda.__file__),
        "selective_scan_binary_sha256": sha256_file(Path(selective_scan_cuda.__file__)),
        "installation": "official 2.2.6.post3 source compiled locally against unchanged torch 2.5.1+cu124",
        "isolated_compiler_prefix": "/home/lrj/.local/cuda-toolchains/cuda-12.4",
    })
    compiler = Path(report["isolated_compiler_prefix"]) / "bin/nvcc"
    report["nvcc"] = subprocess.check_output([str(compiler), "-V"], text=True).strip()
    assert report["output_shape"] == [4, 15, 128]
    assert report["output_finite"] and report["input_gradient_finite"] and report["parameter_gradients_finite"]
    write_json(RESULTS_ROOT / "environment_smoke.json", report)
    print(x.shape, y.shape, report["output_finite"], flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
