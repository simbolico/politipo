"""Utility functions for working with Pydantic."""
from functools import lru_cache


@lru_cache(maxsize=1)
def get_pydantic_version() -> int:
    """
    Get the major version of Pydantic installed.

    Returns:
        int: 1 for Pydantic v1.x, 2 for Pydantic v2.x

    Raises:
        ImportError: If Pydantic is not installed
    """
    try:
        import pydantic
        version = getattr(pydantic, "__version__", "1.0.0")
        return int(version.split(".")[0])
    except ImportError:
        raise ImportError("Pydantic is required but not installed") 