# setup.py

from setuptools import setup, find_packages

setup(
    name='ml_validator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
    ],
)

