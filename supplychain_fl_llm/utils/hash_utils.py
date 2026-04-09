from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np


def _normalize_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    return np.ascontiguousarray(arr)


def hash_ndarrays(arrays: Iterable[np.ndarray]) -> str:
    """Create a deterministic SHA-256 hash for a list of model arrays."""
    hasher = hashlib.sha256()
    for idx, array in enumerate(arrays):
        arr = _normalize_array(array)
        metadata = json.dumps(
            {
                "index": idx,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            },
            sort_keys=True,
        ).encode("utf-8")
        hasher.update(metadata)
        hasher.update(arr.tobytes(order="C"))

    return hasher.hexdigest()


def verify_ndarray_hash(arrays: Iterable[np.ndarray], expected_hash: str) -> bool:
    return hash_ndarrays(arrays) == expected_hash


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
