try:
    from .types import PolarsTypeSystem
    __all__ = ["PolarsTypeSystem"]
except ImportError:
    # Allow importing __init__ even if polars isn't installed
    __all__ = [] 