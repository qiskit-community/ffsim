# Developer guide

The instructions on this page won't work natively on Windows. For ffsim development on Windows, we recommend using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/) and doing development within the WSL.

## Setup

To set up ffsim for development, install it from source in editable mode along with the development requirements:

```bash
pip install -e . --group dev
```

To install the git pre-commit hooks, run

```bash
pre-commit install
```

## Rust

If you add or modify any Rust modules, rebuild them by running the command

```bash
maturin develop
```

If you are benchmarking the code, then pass the `--release` flag:

```bash
maturin develop --release
```

## Run code checks using tox

You can run tests and other code checks using [tox](https://tox.wiki/en/latest/).
To run all checks, simply run

```bash
tox
```

To run a specific check, run

```bash
tox run -e <environment name>
```

substituting `<environment name>` with the name of the tox environment for the check. The following environments are available:

- `test-3.10`, `test-3.11`, `test-3.12`, `test-3.13`, `test-3.14`: Run tests for a specific Python version
- `coverage`: Code coverage
- `type`: Type check
- `lint`: Lint check
- `format`: Format check
- `spell`: Spell check
- `docs`: Build documentation

## Run code checks directly

Running the code checks directly using the corresponding software tool gives you more flexibility and allows you to automatically fix lint and formatting errors.

### Run tests

```bash
pytest
```

### Run type check

```bash
mypy
```

### Fix lint errors

```bash
ruff check --fix
```

### Fix formatting errors

```bash
ruff format
```

### Run spell check

```bash
typos
```

### Build documentation

```bash
python docs/generate_api_docs.py
sphinx-build -b html -W docs/ docs/_build/html
```

## View locally built documentation

After building the documentation, open the file `docs/_build/html/index.html` in your web browser.
