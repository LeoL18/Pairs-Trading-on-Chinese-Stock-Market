"""Package initialization for the src package."""

__all__ = ["__version__", "info"]

__version__ = "0.1.0"


def info():
    """Return package metadata."""
    return {
        "name": __name__,
        "version": __version__,
    }
