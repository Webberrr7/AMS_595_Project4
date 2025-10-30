# markov chain.py
# Implements a 5-state Markov chain:
# 1) Build a random row-stochastic transition matrix P
# 2) Build a random probability vector p and iterate p_{k+1} = P^T p_k for 50 steps
# 3) Compute the stationary distribution as the eigenvector of P^T for eigenvalue ~ 1
# 4) Compare p_50 with the stationary distribution within 1e-5

from __future__ import annotations
import numpy as np

def normalize_rows(P: np.ndarray) -> np.ndarray:
    """
    Row-normalize a matrix so each row sums to 1.
    """
    row_sums = P.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero: if a row is all zeros, replace by uniform row
    zero_rows = (row_sums == 0.0).flatten()
    if np.any(zero_rows):
        P[zero_rows] = 1.0
        row_sums = P.sum(axis=1, keepdims=True)
    return P / row_sums

def normalize_vector(p: np.ndarray) -> np.ndarray:
    """
    Normalize a vector so its entries sum to 1.
    """
    s = p.sum()
    if s == 0.0:
        # If all zeros, fall back to a uniform distribution
        return np.full_like(p, 1.0 / p.size)
    return p / s

def iterate_markov(P: np.ndarray, p0: np.ndarray, steps: int) -> np.ndarray:
    """
    Apply p_{k+1} = P^T p_k repeatedly.
    """
    PT = P.T
    p = p0.copy()
    for _ in range(steps):
        p = PT @ p
    return p

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute a stationary distribution v for P by solving P^T v = v.
    Numerically: take the eigenvector of P^T whose eigenvalue is closest to 1,
    then rescale to a probability vector. We take real parts to remove tiny
    imaginary numerical noise.
    """
    PT = P.T
    evals, evecs = np.linalg.eig(PT)
    # Index of eigenvalue closest to 1
    idx = np.argmin(np.abs(evals - 1.0))
    v = np.real(evecs[:, idx])  # drop small imaginary parts
    # Make all entries nonnegative
    # If most mass is negative, flip sign.
    if v.sum() < 0:
        v = -v
    # If some entries are negative due to numerical noise, clamp at 0 and renormalize
    v = np.maximum(v, 0.0)
    v = normalize_vector(v)
    return v

def main(seed: int = 0, n: int = 5, steps: int = 50, tol: float = 1e-5) -> None:
    np.random.seed(seed)

    # Random transition matrix P, row-stochastic
    P = np.random.rand(n, n)
    P = normalize_rows(P)

    # Random initial probability vector p, normalized
    p0 = np.random.rand(n)
    p0 = normalize_vector(p0)

    # Apply the transition rule 50 times to obtain p_50
    p50 = iterate_markov(P, p0, steps)

    #  Stationary distribution via eigen-decomposition of P^T
    v = stationary_distribution(P)

    # Component-wise difference
    diff = np.abs(p50 - v)

    np.set_printoptions(precision=6, suppress=True)
    print("Transition matrix P (row-stochastic):\n", P, "\n")
    print("Initial distribution p0:\n", p0, "\n")
    print(f"p_{steps} after {steps} steps:\n", p50, "\n")
    print("Stationary distribution v:\n", v, "\n")
    print("Component-wise absolute difference |p_50 - v|:\n", diff, "\n")
    print(f"Max difference: {diff.max():.3e}")
    print(f"Match within {tol}?  ->  {np.all(diff <= tol)}")

if __name__ == "__main__":
    main()
