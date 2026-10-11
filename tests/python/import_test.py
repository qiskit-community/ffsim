# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for importing ffsim with optional dependencies."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

import ffsim


def test_import_ffsim_does_not_import_qiskit():
    """Test that importing ffsim does not import Qiskit."""
    code = "import sys, ffsim; assert 'qiskit' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dir_includes_qiskit_without_importing_it():
    """Test that dir(ffsim) lists Qiskit without importing it."""
    code = (
        "import sys, ffsim; "
        "assert 'qiskit' in dir(ffsim); "
        "assert 'qiskit' not in sys.modules; "
        "assert 'ffsim.qiskit' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_attribute():
    """Test that accessing a missing attribute raises AttributeError."""
    with pytest.raises(AttributeError, match="no_such_attribute"):
        _ = ffsim.no_such_attribute  # type: ignore[attr-defined]


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is not None,
    reason="Qiskit is installed",
)
def test_missing_qiskit():
    """Test that missing Qiskit produces an error with installation instructions."""
    with pytest.raises(ImportError, match=r'pip install "ffsim\[qiskit\]"'):
        _ = ffsim.qiskit
