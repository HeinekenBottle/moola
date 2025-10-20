"""Backward compatibility shim for data_validation module.

This module re-exports DataValidator from its new location at
utils/validation/data_validation.py to maintain backward compatibility.
"""

from .validation.data_validation import DataValidator

__all__ = ["DataValidator"]
