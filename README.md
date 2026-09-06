# Coursera notebooks

Course notebooks from the Coursera *TensorFlow: Advanced Techniques* specialization, together with PyTorch ports of them.

| Folder | Contents |
| --- | --- |
| [`cv/`](cv/) | The original TensorFlow/Keras notebooks for course 3, *Advanced Computer Vision with TensorFlow* — 16 labs and graded assignments, unmodified. |
| [`cv_pytorch/`](cv_pytorch/) | PyTorch ports of all 16, plus a prerequisites notebook with no TensorFlow counterpart. See [`cv_pytorch/README.md`](cv_pytorch/README.md) for the per-notebook notes. |

Each port keeps the structure, prose and exercises of the original; only the framework changed. The Coursera autograders expect Keras artifacts, so the PyTorch assignments cannot be submitted for grading — they are for working through the material.

## Running the PyTorch notebooks

Each ported course is its own [uv](https://docs.astral.sh/uv/) project.

```bash
cd cv_pytorch
uv sync                 # creates .venv and installs everything from uv.lock
uv run jupyter lab
```

Training uses CUDA when available, Apple's MPS backend on Apple Silicon Macs, and the CPU otherwise.

## What is not in the repo

Datasets, downloaded sample images, pretrained weights (`*.pt`, `*.h5`) and notebook output artifacts are ignored — the notebooks re-download or regenerate them on first run. Only the notebooks and the environment files (`pyproject.toml`, `uv.lock`) are tracked, so a clone is small.
