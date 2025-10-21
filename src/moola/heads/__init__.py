"""Head modules for multi-task learning.

Classification and pointer regression heads.
"""

from .pointer_head import PointerHead, TypeHead

__all__ = [
    "PointerHead",
    "TypeHead",
]
