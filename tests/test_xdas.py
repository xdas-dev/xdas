import re

import xdas as xd

# Release segment, optionally followed by a PEP 440 pre-release marker (e.g. 0.2.8rc0).
VERSION_PATTERN = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?$")


def test_version():
    version = xd.__version__
    assert isinstance(version, str)
    assert VERSION_PATTERN.match(version)
