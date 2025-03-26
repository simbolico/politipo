try:
    from .types import SQLModelTypeSystem
    __all__ = ["SQLModelTypeSystem"]
except ImportError:
    # Allow importing __init__ even if sqlmodel isn't installed
    __all__ = [] 