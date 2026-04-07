# Installation

ffsim is supported directly on Linux and macOS.

ffsim is not supported directly on Windows. Windows users have two main options:

- Use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/). WSL provides a Linux environment where ffsim can be pip installed from PyPI or from source.
- Use ffsim within Docker. See [Use within Docker](#use-within-docker).

## Pip install

ffsim is available on [PyPI](https://pypi.org/project/ffsim/). It can be installed by running

```bash
pip install ffsim
```

## Install from source

You can use pip to install ffsim from source. For example:

```bash
git clone https://github.com/qiskit-community/ffsim.git
cd ffsim
pip install .
```

## Use within Docker

We have provided a [Dockerfile](https://github.com/qiskit-community/ffsim/blob/main/Dockerfile), which can be used to build a [Docker](https://www.docker.com/) image, as well as a [compose.yaml](https://github.com/qiskit-community/ffsim/blob/main/compose.yaml) file, which allows one to use the Docker image with just a few simple commands:

```bash
git clone https://github.com/qiskit-community/ffsim.git
cd ffsim
docker compose build
docker compose up
```

Depending on your system configuration, you may need to type `sudo` before each `docker` command.

Once the container is running, navigate to <http://localhost:58888> in a web browser to access the Jupyter Notebook interface.

The home directory includes a subdirectory named `persistent-volume`. All work you’d like to save should be placed in this directory, as it is the only one that will be saved across different container runs.
