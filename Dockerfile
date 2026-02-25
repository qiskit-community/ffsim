FROM quay.io/jupyter/minimal-notebook:python-3.13

LABEL maintainer="Kevin J. Sung <kevinsung@ibm.com>"

# The base notebook sets up a `work` directory "for backwards
# compatibility".  We don't need it, so let's just remove it.
RUN rm -rf work

# Install apt dependencies
USER root
RUN apt update && apt install -y libssl-dev libopenblas-dev pkg-config
USER ${NB_UID}

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/home/jovyan/.cargo/bin:${PATH}"

# Copy files
COPY . .src/ffsim

# Fix the permissions of ~/.src and ~/persistent-volume
USER root
RUN fix-permissions .src && \
    mkdir persistent-volume && fix-permissions persistent-volume
USER ${NB_UID}

# Consolidate the docs into the home directory
RUN mkdir docs && \
    cp -a .src/ffsim/docs docs/ffsim

# Pip install ffsim
RUN pip install -e ".src/ffsim[dev]"
