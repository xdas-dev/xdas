import os


class Config:
    config = {"n_workers": os.cpu_count()}


def get(key):
    return Config.config[key]


def set(key, value):
    Config.config[key] = value
