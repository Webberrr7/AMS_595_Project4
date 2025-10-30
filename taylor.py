# taylor.py
# Implements:
# 1: Truncated Taylor approximation for an arbitrary analytic function
# 2: Example with f(x)=x*sin(x)^2 + cos(x) on [-10,10], 100 points, m=99, c=0
# 3: Sweep degrees (m) and record L1 error sum and runtime; save to CSV

from __future__ import annotations
import time
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Symbol, diff, lambdify, Expr, sin, cos


def taylor_coefficients(
    f_expr: Expr, x: Symbol, c: float, degree: int
) -> np.ndarray:
    """
    Compute coefficients a_n = f^{(n)}(c) / n! for n=0..degree.
    Returns a NumPy array of shape (degree+1,).
    """
    from math import factorial

    coeffs = []
    f_n = f_expr
    for n in range(degree + 1):
        # Evaluate n-th derivative at x=c
        fn_c = float(f_n.subs(x, c))
        coeffs.append(fn_c / factorial(n))
        # Prepare next derivative
        f_n = diff(f_n, x)
    return np.array(coeffs, dtype=float)


def taylor_evaluate(
    coeffs: np.ndarray, x_points: np.ndarray, c: float
) -> np.ndarray:
    """
    Evaluate the truncated Taylor polynomial at x_points using Horner's method
    on (x - c). Works vectorized for NumPy arrays.
    p(x) = sum_{n=0}^m a_n (x-c)^n
    """
    y = np.zeros_like(x_points, dtype=float)
    # Horner on ascending powers: y = (((a_m)*(x-c) + a_{m-1})*(x-c) + ...) + a_0
    for a in coeffs[::-1]:
        y = y * (x_points - c) + a
    return y


def taylor_approximation(
    f_expr: Expr,
    start: float,
    end: float,
    degree: int,
    fixed_c: float,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Q3.1 — Build a truncated Taylor approximation on [start, end].
    Returns (x_grid, fhat) where fhat are the approximated values.
    """
    x = Symbol("x", real=True)
    # grid
    xs = np.linspace(start, end, num_points)

    # coefficients a_n = f^{(n)}(c)/n!
    coeffs = taylor_coefficients(f_expr, x, fixed_c, degree)

    # evaluate polynomial on grid
    fhat = taylor_evaluate(coeffs, xs, fixed_c)
    return xs, fhat


def sweep_degrees(
    f_expr: Expr,
    start: float,
    end: float,
    fixed_c: float,
    initial_degree: int,
    final_degree: int,
    degree_step: int,
    num_points: int = 100,
) -> pd.DataFrame:
    """
    Run Taylor approximation for m = initial_degree, initial_degree+degree_step, ..., final_degree.
    For each m:
      - compute sum_i |f(x_i) - fhat(x_i)|
      - measure runtime (seconds)
    Returns a DataFrame with columns: ['degree', 'l1_error_sum', 'seconds'].
    """
    x = Symbol("x", real=True)
    f_num = lambdify(x, f_expr, modules="numpy")
    xs = np.linspace(start, end, num_points)

    rows = []
    for m in range(initial_degree, final_degree + 1, degree_step):
        t0 = time.perf_counter()
        _, fhat = taylor_approximation(
            f_expr=f_expr,
            start=start,
            end=end,
            degree=m,
            fixed_c=fixed_c,
            num_points=num_points,
        )
        sec = time.perf_counter() - t0

        # true function values
        f_true = f_num(xs)
        l1_err = float(np.sum(np.abs(f_true - fhat)))

        rows.append({"degree": m, "l1_error_sum": l1_err, "seconds": sec})

    return pd.DataFrame(rows)



def plot_example(
    xs: np.ndarray, f_true: np.ndarray, f_hat: np.ndarray, out_png: str
) -> None:
    """
    Plot Actual vs Taylor Approximation and save the figure.
    """
    plt.figure()
    plt.plot(xs, f_hat, marker="o", linestyle="-", label="Taylor Approximation")
    plt.plot(xs, f_true, linestyle="-", label="Actual")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Example Taylor series approximation")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()


def main() -> None:
    # Target function for Q3.2:
    # f(x) = x*sin(x)^2 + cos(x)
    x = Symbol("x", real=True)
    f_expr = x * sin(x) ** 2 + cos(x)

    # Q3.2 parameters
    start, end = -10.0, 10.0
    num_points = 100
    degree = 99   # m = 99 => 100 terms 0..99
    c0 = 0.0

    # Build numerical callable for true values
    f_num = lambdify(x, f_expr, modules="numpy")

    # Compute Taylor approximation and plot
    xs, fhat = taylor_approximation(
        f_expr=f_expr,
        start=start,
        end=end,
        degree=degree,
        fixed_c=c0,
        num_points=num_points,
    )
    ftrue = f_num(xs)

    # Plot and save figure
    plot_example(xs, ftrue, fhat, out_png="taylor_example.png")

    # Q3.3: sweep degrees and save CSV
    df = sweep_degrees(
        f_expr=f_expr,
        start=start,
        end=end,
        fixed_c=c0,
        initial_degree=50,
        final_degree=100,
        degree_step=10,
        num_points=num_points,
    )
    print(df)
    df.to_csv("taylor.values.csv", index=False)
    print("Saved metrics to taylor.values.csv")


if __name__ == "__main__":
    main() 
