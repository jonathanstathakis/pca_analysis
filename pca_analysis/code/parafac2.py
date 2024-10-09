from dataclasses import dataclass
import numpy as np
from tensorly.parafac2_tensor import apply_parafac2_projections

@dataclass
class Pure:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray

@dataclass
class Scaled:
    B: np.ndarray
    C: np.ndarray


class FactorChecker:
    def __init__(self, I, J, K):
        """
        Validate the factor matrices against the shapes of X.

        Stores the expected proportions
        """

@dataclass
class Modes:
    I: int
    J: int
    K: int


class Parafac2:

    def __init__(self, decomposition, X):
        """
        A contains the pure concentrations
        B contains the pure elution profiles
        C contains the pure spectral profiles

        B and C must be scaled by corresponding A to match X
        """
        self.X = X
        self.X_modes = Modes(*self.X.shape)
        
        self.weights, (A, B, C), self._projections = decomposition
        
        _, (_, B_, _) = apply_parafac2_projections((self.weights, (A, B, C), self._projections))
        
        self.pure = Pure(A, B_, C)
