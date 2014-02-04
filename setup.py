#!/usr/bin/python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension
except ImportError :
    raise ImportError("setuptools module required, please go to https://pypi.python.org/pypi/setuptools and follow the instructions for installing setuptools")
import sys

setup(
    name='wrappers',
    url='https://github.com/schaunwheeler/wrappers',
    version='0.1',
    packages=['wrappers'],
    license='The MIT License: http://www.opensource.org/licenses/mit-license.php',
    install_requires=['numpy', 'fastcluster', 'hcluster', 'networkx', 'dedupe', 'pandas', 'sklearn', 'multiprocessing'],
    long_description=open('README.md').read(),
    )
