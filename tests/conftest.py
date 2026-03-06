"""
pytest configuration for distributed-ai tests.

Automatically skips the whole suite if PyTorch is not installed so the repo
can still be cloned and inspected on machines without GPU dependencies.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_torch: mark tests that need PyTorch installed",
    )


def pytest_collection_modifyitems(config, items):
    try:
        import torch  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="PyTorch not installed — pip install torch")
        for item in items:
            item.add_marker(skip)
