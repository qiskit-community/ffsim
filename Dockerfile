FROM quay.io/jupyter/minimal-notebook:python-3.11

LABEL maintainer="Kevin J. Sung <kevinsung@ibm.com>"

# The base notebook sets up a `work` directory "for backwards
# compatibility".  We don't need it, so let's just remove it.
RUN rm -rf work

# Install apt dependencies
USER root
RUN apt update && apt install -y libssl-dev rustc cargo libopenblas-dev pkg-config
USER ${NB_UID}

# Copy files
COPY docs docs/
RUN mkdir .src
COPY python .src/ffsim/python/
COPY src .src/ffsim/src/
COPY tests .src/ffsim/tests/
COPY Cargo.lock Cargo.toml LICENSE pyproject.toml README.md .src/ffsim/

# Fix file permissions
USER root
RUN fix-permissions docs && fix-permissions .src && \
    mkdir persistent-volume && fix-permissions persistent-volume
USER ${NB_UID}

# Pip install ffsim
RUN pip install -e .src/ffsim
