# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test that no warnings are raised in Python files."""

import warnings
from pathlib import Path

import pytest


def test_no_warnings():
    """Test that no warnings are raised in Python files."""
    all_warnings = []
    for filepath in Path("python").glob("**/*.py"):
        with warnings.catch_warnings(record=True) as caught_warnings:
            compile(filepath.read_text(), filepath, "exec")
            all_warnings.extend(caught_warnings)
    if all_warnings:
        error_messages = [
            f"{w.filename}:{w.lineno}: {w.category.__name__}: {w.message}"
            for w in all_warnings
        ]
        pytest.fail("Encountered warnings:\n" + "\n".join(error_messages))
