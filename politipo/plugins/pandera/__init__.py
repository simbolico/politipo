"""Pandera plugin for Politipo."""

try:
    import pandera as pa
    import pandas as pd
    import numpy as np
    from .types import PanderaTypeSystem
    __all__ = ["PanderaTypeSystem"]
except ImportError as e:
    # Allow importing __init__ even if dependencies aren't installed
    __all__ = []
    # Optionally log which dependency is missing
    import warnings
    warnings.warn(f"Pandera plugin dependencies not satisfied: {e}. The plugin will be disabled.") 