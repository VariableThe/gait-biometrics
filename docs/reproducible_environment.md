# Locked environments and CPU checks

The project pins Python 3.14.8 and the existing runtime requirements in
`pyproject.toml`. `uv.lock` also pins transitive packages and development tools.
The original requirements files remain supported for existing environments.

```bash
# Isolated, CPU-only environment; no CUDA libraries needed for the tests.
uv sync --frozen --extra cpu --compile-bytecode
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' uv run --frozen --extra cpu python -m pytest -q
uv audit --frozen
uv run --frozen --extra cpu pip-audit --local
uv run --frozen --extra cpu pip-audit -r requirements.txt --no-deps --disable-pip --timeout 60

# NVIDIA CUDA 13.0 wheels on a compatible Linux/Windows GPU machine.
uv sync --frozen --extra cu130 --compile-bytecode
uv run --frozen --extra cu130 python -m scripts.run_experiments --device cuda
```

Choose one backend extra consistently when syncing or running. CPU and CUDA
extras conflict intentionally. Without an extra, the existing PyPI torch wheel
selection applies. CUDA needs a compatible device/driver; successful CPU checks
do not establish GPU training determinism. Other Python minor versions and
operating systems have not been validated by this follow-up.

The GitHub workflow uses the CPU extra, frozen lock, pinned action commits and
single-thread tests. Pinned Ruff and ty check the Python surfaces listed in
`.github/python-quality-files.txt`; this is an explicit scope for the follow-up,
not a claim that every legacy file has been migrated to strict typing.
It downloads dependencies, but no datasets and no training
artifacts. Package-cache reuse avoids repeated wheel downloads. The local
integration check uses a separate environment so it cannot mutate an active
training environment. No startup or installation speedup is claimed.

Backend configuration follows the [official uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
Vulnerability results are time-dependent; the workflow audits on each run.
`pip-audit --local` cannot resolve torch's `+cpu` suffix through PyPI's API;
the separate pinned-release requirements audit covers the upstream version.
The lock audit covers all resolved dependencies. None proves the absence of
unknown vulnerabilities.
