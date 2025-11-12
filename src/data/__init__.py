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
from .clean_reviews import clean_reviews_dataset
from .clean_metadata import clean_metadata_dataset
from .fuse_datasets import fuse_reviews_metadata

__all__ = [
    "DATA_DIR",
    "get_data_path",
    "iter_json_chunks",
    "load_sample",
    "DatasetProfile",
    "profile_dataset",
    "clean_reviews_dataset",
    "clean_metadata_dataset",
    "fuse_reviews_metadata",
]

