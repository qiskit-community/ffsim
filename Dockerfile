FROM quay.io/jupyter/minimal-notebook:python-3.13

LABEL maintainer="Kevin J. Sung <kevinsung@ibm.com>"

# The base notebook sets up a `work` directory "for backwards
# compatibility".  We don't need it, so let's just remove it.
RUN rm -rf work

# Install apt dependencies
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        libopenblas-dev \
        libssl-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

# Copy files
COPY . .src/ffsim

# Fix the permissions of ~/.src and ~/persistent-volume
USER root
RUN fix-permissions .src && \
    mkdir persistent-volume && \
    fix-permissions persistent-volume
USER ${NB_UID}

# Consolidate the docs into the home directory
RUN mkdir docs && \
    cp -a .src/ffsim/docs docs/ffsim

# Install ffsim and documentation dependencies
RUN python -m pip install --no-cache-dir --upgrade "pip>=25.1" && \
    python -m pip install --no-cache-dir -e .src/ffsim \
        --group .src/ffsim/pyproject.toml:docs
