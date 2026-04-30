class Engine:
    _registry = {}

    def __init_subclass__(cls, *, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            Engine._registry[name] = cls

    def __class_getitem__(cls, item):
        return cls._registry[item]

    @staticmethod
    def open_dataarray(fname, **kwargs):
        raise NotImplementedError

    @staticmethod
    def save_dataarray(da, fname, **kwargs):
        raise NotImplementedError


# TODO: move engines to related submodules


class APSensingEngine(Engine, name="apsensing"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .apsensing import read

        return read(fname, **kwargs)


class ASNEngine(Engine, name="asn"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .asn import read

        return read(fname, **kwargs)


class FebusEngine(Engine, name="febus"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .febus import read

        return read(fname, **kwargs)


class MiniSEEDEngine(Engine, name="miniseed"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .miniseed import read

        return read(fname, **kwargs)


class NetCDFEngine(Engine, name="netcdf"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .netcdf import open_dataarray

        return open_dataarray(fname, **kwargs)

    @staticmethod
    def save_dataarray(da, fname, **kwargs):
        from .netcdf import save_dataarray

        return save_dataarray(da, fname, **kwargs)

    @staticmethod
    def open_datacollection(fname, **kwargs):
        from .netcdf import open_datamapping

        return open_datamapping(fname, **kwargs)

    @staticmethod
    def save_datacollection(dc, fname, **kwargs):
        from .netcdf import save_datamapping

        return save_datamapping(dc, fname, **kwargs)


class OptaSenseEngine(Engine, name="optasense"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .optasense import read

        return read(fname, **kwargs)


class SilixaEngine(Engine, name="silixa"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .silixa import read

        return read(fname, **kwargs)


class SintelaEngine(Engine, name="sintela"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .sintela import read

        return read(fname, **kwargs)


class Terra15Engine(Engine, name="terra15"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        from .terra15 import read

        return read(fname, **kwargs)
