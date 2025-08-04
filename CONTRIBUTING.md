# Developer guide

The instructions on this page won't work natively on Windows. For ffsim development on Windows, we recommend using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/) and doing development within the WSL.

## Setup

To set up ffsim for development, install it from source in editable mode along with the development requirements:

```bash
pip install -e ".[dev]"
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

- `py38`, `py39`, `py310`, `py311`, `py312`: Run tests for a specific Python version
- `coverage`: Code coverage
- `type`: Type check
- `lint`: Lint check
- `format`: Format check
- `docs`: Build documentation

## Run code checks directly

Running the code checks directly using the corresponding software tool can be useful and allows you to:

- Automatically fix lint and formatting errors.
- Build the documentation without deleting cached files.

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

### Build documentation

```bash
sphinx-build -b html -W docs/ docs/_build/html
```

## View locally built documentation

After building the docs using either the [tox command](#run-code-checks-using-tox) or the [sphinx command](#build-documentation), open the file `docs/_build/html/index.html` in your web browser. For rapid iterations, the sphinx command is preferred because it retains cached files.
Building the documentation can consume significant CPU because the tutorial notebooks are executed.
The tox command deletes cached files so it will execute all the notebooks every time, while the sphinx command only executes notebooks if they were modified from the previous run.
