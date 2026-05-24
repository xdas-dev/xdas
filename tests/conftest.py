import pytest

import xdas


def pytest_configure(config):
    xdas.config.set("n_workers", 1)


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="slow test, skipped with --skip-slow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
