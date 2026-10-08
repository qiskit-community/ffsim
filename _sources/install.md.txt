# Installation

ffsim is supported directly on Linux and macOS.

ffsim is not supported directly on Windows. Windows users have two main options:

- Use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/). WSL provides a Linux environment where ffsim can be pip installed from PyPI or from source.
- Use ffsim within Docker. See [Use within Docker](#use-within-docker).

## Pip install

ffsim is available on [PyPI](https://pypi.org/project/ffsim/). You can install it by running

```bash
pip install ffsim
```

For improved performance on [x86](https://en.wikipedia.org/wiki/X86) systems, considering [installing from source](#install-from-source).

## Install from source

You can use pip to install ffsim from source. For example:

```bash
git clone https://github.com/qiskit-community/ffsim.git
cd ffsim
pip install .
```

Installing from source may improve performance on x86 systems because the Rust extensions in the PyPI wheels are compiled with `-C target-cpu=x86-64`, which targets the baseline x86-64 instruction set for broad compatibility. When you build from source, ffsim is configured to compile its Rust extensions with `-C target-cpu=native`, so the Rust compiler can emit optimized instructions (e.g., AVX2, AVX-512) for your specific CPU.

Similarly, you can install [PySCF](https://pyscf.org/) from source with `-DBUILD_MARCH_NATIVE=ON` to enable CPU-specific optimizations in PySCF's C extensions. See [PySCF's installation instructions](https://pyscf.org/user/install.html#build-from-source) for details.

## Use within Docker

We provide a [Dockerfile](https://github.com/qiskit-community/ffsim/blob/main/Dockerfile) and a [compose.yaml](https://github.com/qiskit-community/ffsim/blob/main/compose.yaml) file, which you can use to build a [Docker](https://www.docker.com/) image with just a few simple commands:

```bash
git clone https://github.com/qiskit-community/ffsim.git
cd ffsim
docker compose build
docker compose up
```

Depending on your system configuration, you may need to type `sudo` before each `docker` command.

Once the container is running, navigate to <http://localhost:58888> in a web browser to access the Jupyter Notebook interface.

The home directory includes a subdirectory named `persistent-volume`. All work you’d like to save should be placed in this directory, as it is the only one that will be saved across different container runs.
