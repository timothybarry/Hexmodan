"""AZMO Mind package.

The version is read from installed package metadata rather than written here.
It used to be a literal, and it drifted: this file said 0.2.5 while
``pyproject.toml`` said 0.2.8 and the docs said 0.2.6. Three numbers, all
quotable, none checked. One source of truth is `pyproject.toml`; everything else
asks.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("azmo-mind")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+source"

__all__ = ["__version__"]
