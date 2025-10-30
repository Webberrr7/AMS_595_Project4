# AMS 595 Python Project 2

## Fractal Approximation, Markov Chains & Taylor Series

This project implements three numerical computation tasks:

1. **Mandelbrot set visualization**
2. **Markov chain stationary distribution computation**
3. **Taylor series approximation for analytic functions**

---

##  Project Files

| File | Description |
|------|------------|
| `mandelbrot.py` | Generates Mandelbrot fractal over [-2, 1] × [-1.5, 1.5] |
| `markov_chain.py` | Simulates 5-state Markov chain and computes stationary distribution |
| `taylor.py` | Computes truncated Taylor series and plots results |
| `taylor.values.csv` | Error & runtime data for Taylor approximation |
| `mandelbrot.png` | Mandelbrot fractal output |
| `taylor_example.png` | Taylor vs Actual output |

---

##  Mandelbrot Set

- Iteration: `z(n+1) = z(n)^2 + c`, `z0 = 0`
- Domain: `[-2,1] × [-1.5,1.5]`
- Threshold: 50  
- Output: `mandelbrot.png`

---

##  Markov Chain

- Random `5x5` row-stochastic matrix
- Random normalized initial distribution
- 50 iterations: `p(k+1) = P^T p(k)`
- Stationary distribution from eigenvector of `P^T`
- Verified `|p50 – v| < 1e-5`

---

## 📈 Taylor Series Approximation

- Target: `f(x) = x sin^2(x) + cos(x)`
- Domain: `[-10,10]`, 100 points
- Expansion point: `c = 0`
- Degree: `m = 99`
- Output: `taylor_example.png`

### Accuracy Sweep

Degrees tested: 50, 60, 70, 80, 90, 100  
Output stored in: `taylor.values.csv`

---

##  Requirements

Install dependencies:

```bash
pip install numpy sympy pandas matplotlib

---

##  Usage

python mandelbrot.py
python markov_chain.py
python taylor.py



