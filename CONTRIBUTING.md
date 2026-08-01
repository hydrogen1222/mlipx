# Contributing to mlipx

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints (`ruff check`).

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Development setup

```bash
# UMA environment (default)
uv sync
uv run mlipx doctor

# Optional engine environments (each isolated, see README)
# .venv-mace / .venv-dpa / .venv-grace
```

Run the test suite with:

```bash
uv run python -m pytest tests/mlipx -q
```

## License

By contributing to mlipx, you agree that your contributions will be licensed
under the MIT License (see `LICENSE.md`). mlipx builds on
[FAIRChem](https://github.com/FAIR-Chem/fairchem) (MIT, Copyright © Meta
Platforms, Inc. and affiliates) -- see `mlipx/LICENSE` for the original
license notice.
