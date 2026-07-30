import re

import xdas as xd

# Release segment, plus the optional PEP 440 pre/post/dev markers (e.g. 0.2.9.dev0).
VERSION_PATTERN = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def test_version():
    version = xd.__version__
    assert isinstance(version, str)
    assert VERSION_PATTERN.match(version)
