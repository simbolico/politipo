"""Utility functions for the politipo core package."""
from politipo.core.utils.pydantic import get_pydantic_version

__all__ = ["get_pydantic_version", "make_hashable", "safe_json_dumps"]

"""
General utilities for the core package.
"""

import json
from typing import Any, Dict, List, Tuple


def make_hashable(value: Any) -> Any:
    """
    Convert a value to a hashable type, useful for dictionaries or objects with
    unhashable components that need to be cached or used as keys.
    
    Args:
        value: Any Python value
        
    Returns:
        A hashable representation of the value
    """
    if isinstance(value, dict):
        # Sort items and convert them to a tuple of tuples
        return tuple(
            (make_hashable(k), make_hashable(v)) 
            for k, v in sorted(value.items())
        )
    elif isinstance(value, list):
        return tuple(make_hashable(x) for x in value)
    elif isinstance(value, set):
        return frozenset(make_hashable(x) for x in value)
    elif hasattr(value, '__dict__'):
        # For objects, try to convert their __dict__ to hashable
        # Exclude methods and private attributes
        try:
            obj_dict = {
                k: v for k, v in value.__dict__.items()
                if not k.startswith('_') and not callable(v)
            }
            return (value.__class__.__name__, make_hashable(obj_dict))
        except (AttributeError, TypeError):
            # Fallback: use str representation
            return str(value)
    else:
        # Basic types are already hashable
        return value


def safe_json_dumps(value: Any) -> str:
    """
    Safely convert any value to a JSON string, handling special objects.
    
    Args:
        value: Any Python value
        
    Returns:
        A JSON string representation
    """
    try:
        return json.dumps(value, default=lambda o: str(o))
    except Exception:
        return str(value) 