"""
Setup script for hnswdb.

All package metadata lives in pyproject.toml. This file only exists to
register the CFFI extension, which compiles the C core during installation.
"""

from setuptools import setup

setup(
    cffi_modules=["hnswdb/build_cffi.py:ffi"],
)
