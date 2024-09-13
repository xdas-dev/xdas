import pytest
from dask.utils import itemgetter, methodcaller

import xdas as xd
from xdas.dask.serial import dumps, loads


def test_tuple():
    objs = [
        (1, 2, 3),
        [1, 2, 3],
        (1, (2, 3)),
        [1, [2, 3]],
        (1, [2, 3]),
        [1, (2, 3)],
    ]
    for obj in objs:
        assert loads(dumps(obj)) == obj


def test_slice():
    objs = [
        slice(1, 2, 3),
        slice(None),
    ]
    for obj in objs:
        assert loads(dumps(obj)) == obj


def test_callable():
    objs = [
        dumps,
        loads,
        xd.DataArray,
        xd.open_dataarray,
    ]
    for obj in objs:
        assert loads(dumps(obj)) is obj


def test_keys():
    obj = {("a", 0, 0): ("b", 0, 0)}
    assert loads(dumps(obj)) == obj


def test_mixed_structure():
    obj = {
        "a": (1, 2, 3),
        ("b", 0, 0): [1, 2, 3],
        "c": (None, slice(1, 2, 3), slice(None)),
        ("d", 1, 1): (dumps, "data"),
        "e": (xd.DataArray, "path"),
    }
    assert loads(dumps(obj)) == obj


def test_methdocaller():
    obj = methodcaller("method")
    assert loads(dumps(obj)) == obj


def test_itemgetter():
    obj = itemgetter(1)
    assert loads(dumps(obj)) == obj


def test_unknown_type():
    with pytest.raises(
        TypeError, match="Cannot encode object of type <class 'object'>"
    ):
        dumps(object())
