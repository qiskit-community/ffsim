# ffsim

Faster simulations of fermionic quantum circuits.

## What is ffsim?

<!-- start what-is-ffsim -->

ffsim is a software library for simulating fermionic quantum circuits that conserve particle number
and the Z component of spin. This category includes many quantum circuits used for quantum chemistry simulations.
By exploiting the symmetries and using specialized algorithms, ffsim can simulate these circuits much faster
than a generic quantum circuit simulator.

**Experimental disclaimer**: ffsim is currently an experimental release. Breaking changes may be introduced without warning.

<!-- end what-is-ffsim -->

## Documentation

Documentation is located at the [project website](https://qiskit-community.github.io/ffsim/).

## Supported platforms

ffsim is supported on Linux, macOS, and the Windows Subsystem for Linux (WSL). It is not supported on Windows.

## Installation

### From PyPI

<!-- start install-from-pypi -->

ffsim is available on [PyPI](https://pypi.org/project/ffsim/). It can be installed by running the command

```bash
pip install ffsim
```

<!-- end install-from-pypi -->

### From source

Installing ffsim from source requires the following system dependencies:

- A Rust compiler. See [these instructions](https://www.rust-lang.org/tools/install).
- A BLAS implementation.
  - On macOS, ffsim uses the [Accelerate](https://developer.apple.com/documentation/accelerate) framework that is included with the operating system, so no action is required.
  - On Linux, ffsim uses [OpenBLAS](https://www.openblas.net/). You may be able to install it using your system package manager:
    - Arch Linux:

      ```bash
      sudo pacman -S blas-openblas
      ```

    - Fedora:

      ```bash
      sudo dnf install openblas-devel
      ```

    - Ubuntu:

      ```bash
      sudo apt install libopenblas-dev
      ```

Once these dependencies are satisfied, ffsim can be installed by running the command

```bash
pip install .
```

from the root directory of the code repository.

## Development

### Setup

To set up ffsim for development, install it from source in editable mode along with the development requirements:

```bash
pip install -e ".[dev]"
```

To install the git pre-commit hooks, run

```bash
pre-commit install
```

### Rust

If you add or modify any Rust modules, rebuild them by running the command

```bash
maturin develop
```

If you are benchmarking the code, then pass the `--release` flag:

```bash
maturin develop --release
```

### Run code checks using tox

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

### Run code checks directly

Running the code checks directly using the corresponding software tool directly is also useful, for example, for automatically fixing lint or format errors.

#### Run tests

```bash
pytest
```

#### Fix lint errors

```bash
ruff check --fix
```

#### Fix formatting errors

```bash
ruff format
```

## Cite ffsim

You can cite ffsim using the following BibTeX:

```bibtex
@software{ffsim,
  author = {{The ffsim developers}},
  title = {ffsim},
  url = {https://github.com/qiskit-community/ffsim}
}
```
