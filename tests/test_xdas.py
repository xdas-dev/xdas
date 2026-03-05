import xdas as xd


def test_version():
    version = xd.__version__
    assert isinstance(version, str)
    version_parts = version.split(".")
    for part in version_parts:
        assert part.isdigit()
