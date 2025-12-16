# cuda-test-with-python
Testing PyTorch + TensorFlow with GPU acceleration (when available).

This repo is intentionally small and beginner-friendly: it focuses on a couple of GPU checks and simple training examples.

Notebooks:
- [notebooks/gpu-ops-test.ipynb](notebooks/gpu-ops-test.ipynb) — GPU sanity checks + two TensorFlow training examples
- [notebooks/gpu-datasets-pandas.ipynb](notebooks/gpu-datasets-pandas.ipynb) — pandas dataset workflows + where GPU can help (cuDF optional)

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

Optional (CuPy benchmarks):
- Install the optional GPU array library extra: `uv sync --extra gpu`

## Optional: GPU dataframe acceleration with RAPIDS/cuDF (cudf.pandas)

This repo runs on **CPU by default**.

In [notebooks/gpu-datasets-pandas.ipynb](notebooks/gpu-datasets-pandas.ipynb) there is an *optional* cell that tries to enable the pandas accelerator:

- `cudf.pandas` accelerates *some* pandas operations on the GPU with minimal/no code changes.
- If it is not installed, you will see: `GPU acceleration disabled (cudf.pandas not available).` (this is expected).

Important distinction:
- **CuPy** (`import cupy as cp`) = GPU **NumPy-like arrays** (great for numeric compute).
- **cuDF** (`import cudf`) = GPU **DataFrames** (pandas-like API; often uses CuPy/RMM under the hood).

### Installing RAPIDS/cuDF (recommended: separate environment)

RAPIDS/cuDF is usually installed via **conda/mamba** (especially on Linux/WSL), and it often needs tighter CUDA+driver compatibility than typical PyPI packages.
Because of that, it’s commonly best to use a **separate** environment + Jupyter kernel for RAPIDS.

Example (WSL2/Linux, adjust versions to your machine):

```bash
conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
	"rapids=25.08" "python=3.12" "cuda-version>=12.0,<=12.9"
conda activate rapids
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name rapids --display-name "RAPIDS (Py 3.12)"
```

Example (your working setup: RAPIDS 25.12 + TensorFlow + CUDA-enabled PyTorch in one env):

```bash
conda create -n rapids-25.12 -c rapidsai -c conda-forge \
	rapids=25.12 python=3.12 'cuda-version>=12.2,<=12.9' \
	tensorflow 'pytorch=*=*cuda*'

conda activate rapids-25.12

# Add a Jupyter kernel for this environment
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name rapids-25.12 --display-name "RAPIDS 25.12 (CUDA 12.9)"
```

Then switch the notebook kernel to the RAPIDS kernel and run the notebook.

#### Why PyTorch GPU install can fail in the RAPIDS env

RAPIDS releases are built against a specific CUDA minor version range (e.g. RAPIDS 25.12 typically pulls CUDA **12.9** components such as `libnvjitlink`).

The PyTorch conda GPU meta-package (`pytorch-cuda=12.4`, `12.1`, etc.) pins *different* CUDA minor versions.
If those pins disagree, the conda solver will fail with an error similar to:

- RAPIDS/cuDF requires `libnvjitlink >=12.9.*`
- `pytorch-cuda=12.4` requires `libnvjitlink 12.4.*`

In that case, there is no single environment that satisfies both sets of constraints.

Recommended options:

1) **Keep RAPIDS in its own environment** (best for stability) and use a separate ML environment for PyTorch/TensorFlow GPU.
2) If you only need PyTorch for *CPU* in the RAPIDS env, install **CPU-only PyTorch** (no `pytorch-cuda`), and keep GPU work in the ML env.

(It’s sometimes possible to mix conda RAPIDS with `pip install torch` CUDA wheels in the same env, but it’s a higher-risk setup; prefer separate envs unless you’re ok debugging shared-library version issues.)

Official install guide (conda/pip/docker + compatibility): https://docs.rapids.ai/install/

## GPU vs CPU behavior

- If a compatible NVIDIA GPU is available, the notebook will use it automatically.
- If no GPU is available (or drivers aren’t set up), the notebook should still run and will print that it is using CPU.
- The first notebook “sanity check” cell is written to avoid crashes even when no GPU is present.

## Starting on a new Windows PC

What `uv sync` guarantees:
- A reproducible **Python** environment (packages pinned by `uv.lock`).
- This repo is pinned to **Python 3.12.x** (`>=3.12,<3.13`) because some CUDA-related wheels/tooling lag behind 3.13.

Recommended commands:
- Install Python 3.12 via uv (if needed): `uv python install 3.12`
- Create the env (strict): `uv sync --frozen`

What `uv sync` does *not* guarantee (system dependencies):
- NVIDIA driver setup
- GPU availability/visibility inside your runtime (native Windows vs WSL)
- CUDA Toolkit availability (e.g., `ptxas`, `nvcc`) when required by some TensorFlow/XLA GPU paths

Practical expectations:
- **PyTorch**: will use CUDA if available; otherwise runs on CPU. The matrix-multiply notebook cell auto-selects CPU vs CUDA.
- **TensorFlow**: may run CPU-only on native Windows depending on the installed TensorFlow build; GPU support is typically smoother on Linux/WSL.
- For “same behavior as this repo’s WSL setup”, use **WSL2 + NVIDIA Windows driver with WSL support**, then install CUDA Toolkit *inside WSL*.

## Native Linux

This notebook should work on native Linux as well.

- **CPU-only Linux**: everything should run (PyTorch + TensorFlow fall back to CPU).
- **Linux + NVIDIA GPU**: GPU acceleration requires the NVIDIA driver stack on the host.
- **CUDA Toolkit (`ptxas`, `nvcc`)**: some TensorFlow/XLA GPU paths may require CUDA Toolkit binaries; the training example tries to add `/usr/local/cuda/bin` to the kernel `PATH` when present.

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

