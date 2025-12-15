# cuda-test-with-python
Testing PyTorch + TensorFlow with GPU acceleration (when available).

This repo is intentionally small and beginner-friendly: it focuses on a couple of GPU checks and simple training examples.

## Quick start (uv)

- Create/sync the environment:
	- Ensure Python 3.12 is used (recommended when CUDA wheels aren’t available for 3.13):
		- `uv python install 3.12`
	- `uv sync`
	- For strict reproducibility (fail if `uv.lock` and `pyproject.toml` disagree): `uv sync --frozen`
- Run the notebook kernel in VS Code (Jupyter extension) using the venv created by uv.

Notes:
- This project commits `uv.lock` so dependency resolution is reproducible across machines.
- The project is pinned to **Python 3.12.x** (see `requires-python` in `pyproject.toml` and `.python-version`).
- GPU acceleration depends on **system** prerequisites (drivers/toolkit), not just Python packages.

## GPU vs CPU behavior

- If a compatible NVIDIA GPU is available, the notebook will use it automatically.
- If no GPU is available (or drivers aren’t set up), the notebook should still run and will print that it is using CPU.
- The first notebook “sanity check” cell is written to avoid crashes even when no GPU is present.

### Portability expectations

- Python deps are pinned via `uv.lock` (good portability).
- GPU enablement is not guaranteed by Python deps alone; it also depends on:
	- NVIDIA driver support on the host (Windows driver for WSL2, or Linux driver)
	- WSL2 GPU support (if using WSL)
	- CUDA Toolkit availability in the Linux environment for tools like `ptxas` (used by some TF/XLA paths)

## WSL (Windows Subsystem for Linux) notes

This project started on Windows, but most GPU acceleration for these libraries is much smoother on Linux/WSL.

### Prerequisites

- WSL2 installed
- Recent NVIDIA Windows driver with WSL CUDA support
- Inside WSL, CUDA Toolkit installed (for tools like `ptxas` / `nvcc`)

Why the CUDA Toolkit matters:
- Some TensorFlow GPU/XLA execution paths may compile kernels and call `ptxas`.
- Even if CUDA is installed under `/usr/local/cuda`, VS Code/Jupyter kernels sometimes don’t inherit a PATH that includes `/usr/local/cuda/bin`.
- The training notebook includes a small PATH fix to make `/usr/local/cuda/bin` discoverable inside the kernel process.

### Verify from WSL

- Check CUDA Toolkit binaries:
	- `ls -l /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/nvcc`
- In the notebook output, you should see something like:
	- `ptxas: /usr/local/cuda/bin/ptxas`
	- `GPU Available: [PhysicalDevice(...)]`

## Troubleshooting

### TensorFlow complains about missing `ptxas`

If you see errors like “No PTX compilation provider is available” or “Couldn’t find a suitable version of ptxas”, it usually means:
- CUDA Toolkit isn’t installed in WSL/Linux, or
- It is installed but not on PATH for the Jupyter kernel.

The notebook training example prints whether `ptxas`/`nvcc` are found and attempts to add `/usr/local/cuda/bin` to PATH.

