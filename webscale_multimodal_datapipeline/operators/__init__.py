"""
Operators package for data profiling.

This package contains all operator implementations organized by type:
- Refiners: Enrich records with new information
- Filters: Filter records based on criteria
- Dedups: Remove duplicate records

All operators are automatically registered when this package is imported.
"""

# Import all operator types (this will trigger their registration)
from . import (
    dedup,  # noqa: F401
    filters,  # noqa: F401
    refiners,  # noqa: F401
)

__all__ = []
