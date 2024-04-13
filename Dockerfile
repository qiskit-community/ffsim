FROM quay.io/jupyter/minimal-notebook:python-3.11

LABEL maintainer="Kevin J. Sung <kevinsung@ibm.com>"

# Install apt dependencies
USER root
RUN apt update && apt install -y libssl-dev rustc cargo libopenblas-dev pkg-config
USER ${NB_UID}

# The base notebook sets up a `work` directory "for backwards
# compatibility".  We don't need it, so let's just remove it.
RUN rm -rf work && \
    mkdir .src

COPY docs .src/ffsim/docs/
COPY python .src/ffsim/python/
COPY src .src/ffsim/src/
COPY tests .src/ffsim/tests/
COPY Cargo.lock Cargo.toml LICENSE pyproject.toml README.md .src/ffsim/

# Fix the permissions of ~/.src
USER root
RUN fix-permissions .src
USER ${NB_UID}

# Consolidate the docs into the home directory
RUN mkdir docs && \
    cp -a .src/ffsim/docs .

# Pip install ffsim
RUN pip install -e .src/ffsim
