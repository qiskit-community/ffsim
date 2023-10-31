# ffsim

Faster simulations of fermionic quantum circuits.

## What is ffsim?

ffsim is a software library for simulating fermionic quantum circuits that conserve particle number
and the Z component of spin. This category includes many quantum circuits used for quantum chemistry simulations.
By exploiting the symmetries and using specialized algorithms, ffsim can simulate these circuits much faster
than a generic quantum circuit simulator.

**Experimental disclaimer**: ffsim is currently an experimental release. Breaking changes may be introduced without warning.

## Documentation

Documentation is located at the [project website](https://qiskit-community.github.io/ffsim/).

## Supported platforms

ffsim is supported on Linux, macOS, and the Windows Subsystem for Linux (WSL). It is not supported on Windows.

## Installation

### From PyPI

ffsim is available on [PyPI](https://pypi.org/project/ffsim/). It can be installed by running the command

```bash
pip install ffsim
```

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

To set up ffsim for development, install it from source in editable mode along with the development requirements:

```bash
pip install -e ".[dev]"
```

If you add or modify any Rust modules, rebuild them by running the command

```bash
maturin develop
```

If you are benchmarking the code, then pass the `--release` flag:

```bash
maturin develop --release
```

Tests and other code checks are managed using [tox](https://tox.wiki/en/latest/).
To run the default tox environments, simply run

```bash
tox
```

To run a specific environment, for example, to run the lint checks, do

```bash
tox run -e lint
```

You can also use `pytest` to run the tests directly. For example,

```bash
pytest tests/
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
