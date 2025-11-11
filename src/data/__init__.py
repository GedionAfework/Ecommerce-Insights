"""
Data access utilities for the Ecommerce-Insights project.
"""

from .loaders import (
    DATA_DIR,
    get_data_path,
    iter_json_chunks,
    load_sample,
)
from .profile import DatasetProfile, profile_dataset

__all__ = [
    "DATA_DIR",
    "get_data_path",
    "iter_json_chunks",
    "load_sample",
    "DatasetProfile",
    "profile_dataset",
]

