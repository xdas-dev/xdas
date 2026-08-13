from contextlib import ExitStack

import pytest

import xdas


def pytest_configure(config):
    xdas.config.set("n_workers", 1)


@pytest.fixture
def opened():
    """
    Register a ZMQ endpoint to be closed when the test ends, and return it.

    Endpoints are context managers, but a test that hands one to a thread, or
    lets one raise on the way in, reads better flat than nested. Wrapping the
    call in this keeps the socket from outliving the test either way.
    """
    with ExitStack() as stack:
        yield stack.enter_context


@pytest.fixture
def fake_model():
    """
    Return the :func:`tests.fakemodel.fake_model` factory.

    Import the module directly instead when the models are needed at collection
    time, e.g. to build a :func:`pytest.mark.parametrize` table.
    """
    from tests.fakemodel import fake_model

    return fake_model


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
