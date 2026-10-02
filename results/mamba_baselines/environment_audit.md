# Mamba baseline environment audit

Date: 2026-10-02 (Asia/Shanghai)

## Repository

- Repository: `/home/lrj/ped_intent_project`
- Branch: `feat/intent_guided_mamba`
- Starting HEAD: `cb53ed12693f760bfd08637ae4beabb02bad0055`
- `origin/main` was fetched and matched the starting HEAD before branch creation.
- Worktree was clean before branch creation. Existing tracked models and frozen results remain unchanged; new baseline files were added after the authorized continuation below.

## Python / accelerator

- Python: 3.10.21
- PyTorch: 2.5.1+cu124
- `torch.version.cuda`: 12.4
- `torch.cuda.is_available()`: `True`
- GPU: NVIDIA GeForce RTX 3080
- cuDNN: 90100
- `nvcc`: CUDA compilation tools, release 11.5, V11.5.50
- `mamba_ssm`: initially unavailable; now 2.2.6.post3, built locally (see continuation).

## Installation attempt

Command:

```text
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip install mamba-ssm==2.3.2.post1 --no-build-isolation
```

The initial attempt failed during metadata generation. No Torch/CUDA changes were made in that attempt. Further installation work resumed only after user approval.

## Complete pip error output

```text
Collecting mamba-ssm==2.3.2.post1
  Downloading mamba_ssm-2.3.2.post1.tar.gz (216 kB)
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'error'
  error: subprocess-exited-with-error

  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [20 lines of output]
      /home/lrj/anaconda3/envs/ped_intent/lib/python3.10/site-packages/wheel/bdist_wheel.py:4: FutureWarning: The 'wheel' package is no longer the canonical location of the 'bdist_wheel' command, and will be removed in a future release. Please update to setuptools v70.1 or later which contains an integrated version of the 'bdist_wheel' command.
        warn(


      torch.__version__  = 2.5.1+cu124


      Traceback (most recent call last):
        File "/home/lrj/anaconda3/envs/ped_intent/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/home/lrj/anaconda3/envs/ped_intent/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
        File "/home/lrj/anaconda3/envs/ped_intent/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 175, in prepare_metadata_for_build_wheel
          return hook(metadata_directory, config_settings)
        File "/home/lrj/anaconda3/envs/ped_intent/lib/python3.10/site-packages/setuptools/build_meta.py", line 380, in prepare_metadata_for_build_wheel
          self.run_setup()
        File "/home/lrj/anaconda3/envs/ped_intent/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 170, in <module>
      RuntimeError: mamba_ssm is only supported on CUDA 11.6 and above.  Note: make sure nvcc has a supported version by running nvcc -V.
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> mamba-ssm

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

## Initial decision

Local `nvcc` is 11.5 while the package build requires 11.6 or newer. Per the task constraint, stop installation here; do not change CUDA, PyTorch, or the existing `ped_intent` environment. Mamba GPU smoke test, Mamba models, baseline training, test additions, and commit/push are blocked pending an environment/toolchain decision.

## Authorized continuation (2026-10-02)

The user subsequently approved resolving the toolchain issue and continuing the baseline stage.

- The [official 2.3.2.post1 package](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/setup.py) requires Triton >=3.5.0 and has no Torch 2.5 release wheel. The current Torch 2.5.1 installation requires Triton 3.1.0. A compatible official version is selected instead of changing Torch/Triton.
- The [official 2.2.6.post3 release](https://github.com/state-spaces/mamba/releases/tag/v2.2.6.post3) includes a Python 3.10/Torch 2.5/CUDA 12 wheel, but importing it on this host failed with `GLIBC_2.32 not found`. The host is Ubuntu 20.04 with an older GLIBC.
- A dedicated CUDA compiler prefix was created at `/home/lrj/.local/cuda-toolchains/cuda-12.4`, containing nvcc 12.4.131, cudart/dev 12.4.127, and CUDA CCCL 12.4.127. This does not replace `/usr/local/cuda-11.5` or install an alternative PyTorch.
- Mamba 2.2.6.post3 is built from the official PyPI source with `MAMBA_FORCE_BUILD=TRUE`, `MAX_JOBS=2`, `--no-deps`, and `--no-build-isolation`. Detailed build output is retained locally under the ignored `checkpoints/mamba_baselines/mamba_source_install.log`.
- The local source build succeeded. Built wheel SHA256: `d083e96b89c78138bb23ec3cc72da19b18674ba41edf78ec712e73ac0498a28e`.
- GPU forward/backward smoke passed: input/output `[4,15,128]`, finite output, finite input gradient, and finite gradients for all parameters. Details and the installed CUDA extension hash are saved in `environment_smoke.json`.
- Python 3.10.21, Torch 2.5.1+cu124, Torch CUDA 12.4, Triton 3.1.0, and system nvcc 11.5 remain unchanged.
- The optional `causal-conv1d` package is not installed. Official Mamba uses its supported PyTorch convolution fallback and the locally compiled selective-scan CUDA extension. Timing results characterize this installation, not the fastest possible Mamba installation.
- Full post-installation test suite: **176 passed, 0 failed, 0 skipped**, 60 standard Transformer nested-tensor warnings, 9.45 seconds. All new files also passed `py_compile`.

### Reproduction commands

```bash
CUDA_HOME=/home/lrj/.local/cuda-toolchains/cuda-12.4 \
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 \
PATH="/home/lrj/anaconda3/envs/ped_intent/bin:/home/lrj/.local/cuda-toolchains/cuda-12.4/bin:$PATH" \
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip install \
  --force-reinstall --no-deps --no-build-isolation --no-binary=mamba-ssm \
  mamba-ssm==2.2.6.post3

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lrj/anaconda3/envs/ped_intent/bin/python scripts/check_mamba_environment.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pytest tests -q

# Run from /home/lrj/ped_intent_project; existing completed runs are preserved.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lrj/anaconda3/envs/ped_intent/bin/python scripts/run_mamba_baselines.py
```
