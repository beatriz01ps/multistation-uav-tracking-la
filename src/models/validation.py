"""Validation of numpy vectors/matrices used in the Pydantic models"""

from __future__ import annotations

from typing import Optional

import numpy as np

STATE_DIM = 6  


def coerce_state_vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("state contains non-finite values (nan/inf)")
    return array


def coerce_optional_state_vector(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    return coerce_state_vector(value)

_PSD_EIGENVALUE_TOLERANCE = 1e-8


def coerce_covariance_matrix(value) -> np.ndarray:

    array = np.asarray(value, dtype=float)
    if array.shape != (STATE_DIM, STATE_DIM):
        raise ValueError(f"covariance must have shape ({STATE_DIM},{STATE_DIM}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("covariance contains non-finite values (nan/inf)")
    if not np.allclose(array, array.T, atol=1e-6):
        raise ValueError("covariance must be symmetric")
    if np.any(np.diag(array) < 0):
        raise ValueError("covariance has a negative variance on the diagonal")

    eigenvalues = np.linalg.eigvalsh(array)
    smallest = float(eigenvalues.min())
    if smallest < -_PSD_EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"covariance is not positive semidefinite (smallest eigenvalue={smallest:.3e}); "
            "valid covariance matrices have all eigenvalues >= 0"
        )
    return array


def regularize_for_inversion(matrix: np.ndarray, min_eigenvalue: float = 1e-9) -> np.ndarray:

    """Symmetrizes and enforces a minimum eigenvalue floor"""
    
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest = float(eigenvalues.min())
    if smallest < min_eigenvalue:
        matrix = matrix + (min_eigenvalue - smallest) * np.eye(matrix.shape[0])
    return matrix
