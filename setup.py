from setuptools import setup

setup(
    name="igdiff",
    packages=[
        'data',
        'analysis',
        'model',
        'experiments',
        'openfold'
    ],
    package_dir={
        'data': './data',
        'analysis': './analysis',
        'model': './model',
        'experiments': './experiments',
        'openfold': './openfold',
    },
)