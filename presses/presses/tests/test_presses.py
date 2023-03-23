"""
Unit and regression test for the presses package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import presses


def test_presses_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "presses" in sys.modules
