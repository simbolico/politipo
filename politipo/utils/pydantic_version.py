from functools import lru_cache
import sys


@lru_cache(maxsize=1)
def get_pydantic_version() -> int:
    """
    Detects the installed Pydantic major version (1 or 2).

    Returns:
        int: Major version number (1 or 2).

    Raises:
        ImportError: If Pydantic is not installed.
        RuntimeError: If version cannot be determined.
    """
    if 'pydantic' not in sys.modules:
        try:
            import pydantic
        except ImportError:
            # Raise specific error if Pydantic is needed but not found
            raise ImportError("Pydantic is required but not installed.")

    try:
        import pydantic

        # Pydantic v2 has VERSION attribute
        if hasattr(pydantic, "VERSION"):
            version_str = str(pydantic.VERSION)
            if version_str.startswith("2."):
                return 2
            elif version_str.startswith("1."):
                return 1
        # Fallback for older Pydantic v1 or unexpected setups
        elif hasattr(pydantic.BaseModel, "model_validate"):  # v2 method
            return 2
        elif hasattr(pydantic.BaseModel, "dict"):  # v1 method
            return 1

        raise RuntimeError("Cannot determine Pydantic version.")

    except ImportError:
        # This case should ideally be handled before calling,
        # but included for robustness.
        raise ImportError("Pydantic is required but not installed.")
    except Exception as e:
        raise RuntimeError(f"Error determining Pydantic version: {e}") 