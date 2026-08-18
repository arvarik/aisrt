"""Hardware-aware, concurrent pipeline for broadcast-quality subtitle generation."""

try:
    # Written by hatch-vcs at build time from the git tag history.
    from aisrt._version import __version__
except ImportError:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("aisrt")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
