# Conditioning & Stability: When Correct Math Gives Wrong Answers

> A formula can be mathematically perfect and still return garbage on a computer. This project measures *why*, and shows how to tell a hard problem apart from a bad algorithm.

| | |
|---|---|
| **Concepts** | Condition numbers, singular values, floating-point error, eigenvalue sensitivity, least squares, QR decomposition, forward error |
| **Stack** | NumPy, SciPy, SymPy, Matplotlib |
| **Code** | [`conditioning-stability.py`](conditioning-stability.py) |

---

## Why it's cool

Computers store numbers to about 16 significant digits. Most of the time that's plenty, until it isn't. Numerical analysts ask two different questions when results go wrong:

1. **Conditioning**, a property of the *problem*: if the inputs wiggle slightly, how much can the true answer change? Some problems are so sensitive that no algorithm can save you.
2. **Stability**, a property of the *algorithm*: does this method add its own error on top of what the problem forces?

This project shows both, with experiments where the error is easy to see:
- Polynomial roots that move by several units when the coefficients change by one part in ten billion.
- A textbook formula whose error grows to $10^{49}$.

## Where it's used in the real world

- **Machine learning:** ill-conditioned feature matrices make regression unstable. This is why we standardize features and use regularization (ridge regression).
- **Engineering and physics simulation:** the condition number of a stiffness matrix tells you how many digits of a finite-element solution to trust.
- **Control systems and signal processing:** eigenvalue sensitivity determines whether a filter or controller stays stable in practice.
- **Finance:** covariance matrices in portfolio optimization are often near-singular, which leads to wildly unstable "optimal" portfolios.
- **Scientific software:** LAPACK, NumPy, and MATLAB default to QR and SVD solvers for exactly the reasons shown here.

## The concept in 60 seconds

- **Condition number of a matrix:** $\kappa(A) = \sigma_{\max}/\sigma_{\min}$, the ratio of the largest to smallest singular value.
  - $\kappa = 1$ is perfect; orthonormal matrices achieve it.
  - $\kappa = \infty$ means the matrix is singular.
  - Rule of thumb: you lose about $\log_{10}\kappa$ digits of accuracy.
- **Eigenvalue sensitivity:** for **symmetric** (normal) matrices, eigenvalues are well-conditioned. For **non-normal** matrices, a tiny perturbation can move eigenvalues a lot.
- **Least squares:**
  - The *normal equations* $A^\mathsf{T}A\mathbf{x} = A^\mathsf{T}\mathbf{b}$ **square** the condition number.
  - A *QR decomposition* solves $R\mathbf{x} = Q^\mathsf{T}\mathbf{b}$ and keeps $\kappa(A)$ as it is.
  - Same problem, same math, very different accuracy.

## What's implemented

| Function | What it does |
|---|---|
| `matrix_cond(A)` | Condition number from singular values. Tested on orthonormal matrices ($\kappa = 1$) and singular matrices ($\kappa = \infty$), and matches `np.linalg.cond`. |
| `prob2()` | **Wilkinson's polynomial** $\prod_{i=1}^{20}(x - i)$: multiplies each coefficient by a random factor $\sim\mathcal{N}(1, 10^{-10})$ 100 times, plots where the roots go, and estimates absolute and relative condition numbers |
| `eig_cond(A)` | Estimates absolute and relative condition numbers of the eigenvalue problem through random complex perturbations |
| `prob4(domain, res)` | Heat map of the relative eigenvalue condition number of $\begin{bmatrix}1 & x\\ y & 1\end{bmatrix}$ over an $(x, y)$ grid. The matrix is normal when $\lvert x\rvert = \lvert y\rvert$, so conditioning should be good along both diagonals and poor near the axes. |
| `prob5(n)` | Degree-$n$ least-squares polynomial fit, solved with the **normal equations** and with **QR**, plotted together with the residual of each |
| `prob6()` | Computes $I(n) = \int_0^1 x^n e^{x-1}\,dx$ with an exact closed-form identity using subfactorials, and plots the relative error on a log scale |

## Results

**Wilkinson's polynomial.** Its roots are exactly 1, 2, …, 20. Perturbing the coefficients by about $10^{-10}$ (relative) moved the worst root by a **median of 4.9 units**, with roots scattering into the complex plane. Input error was amplified billions of times, which is extreme ill-conditioning. And `np.roots` is a perfectly stable algorithm: the problem itself is the issue.

**Normal equations vs. QR.** Residual $\lVert A\mathbf{x} - \mathbf{b}\rVert_2$ on synthetic noisy data. The repository's `stability_data.npy` isn't included, so the same code was run on generated data.

| Polynomial degree | Normal equations | QR |
|---|---|---|
| 8 | 2.577 | 2.577 |
| 14 | **14.51** | 2.458 |
| 20 | **17.54** | 2.418 |

At low degree the two match. As the Vandermonde matrix becomes ill-conditioned, the normal equations fall apart while QR keeps improving.

**Correct formula, unstable computation.** The subfactorial formula for $I(n)$ is algebraically exact, but it subtracts two huge, nearly equal numbers:

| $n$ | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| Relative error | 1e-14 | 6e-11 | 1e+00 | 6e+17 | **8e+49** |

By $n = 20$ every digit is wrong. This is called **catastrophic cancellation**.

## Run it

```bash
cd Conditional_Stability
python -i conditioning-stability.py
```

```python
>>> prob2()          # Wilkinson root scatter plot
>>> prob4(res=200)   # eigenvalue conditioning heat map
>>> prob6()          # catastrophic cancellation plot
```

`prob5` needs `stability_data.npy`, which isn't included in the repository; any `(x, y)` data on $[0, 1]$ saved with `np.save` works. `test_matrix_cond()` checks `matrix_cond` against `np.linalg.cond` and passes.

## Skills demonstrated

- Deep understanding of floating-point arithmetic and of which numerical results can be trusted.
- Choosing algorithms by stability, not just correctness (QR over the normal equations).
- Designing experiments that isolate and quantify numerical error.
- Communicating subtle technical ideas through visualization.

## Known gap (a good exercise)

`eig_cond` subtracts the original and perturbed eigenvalues **position by position**, but `scipy.linalg.eig` doesn't promise to return them in the same order. When the order flips, the estimate jumps by about $10^{11}$. For example, it reports $2.9\times10^{11}$ for the symmetric matrix $\begin{bmatrix}1&50\\50&1\end{bmatrix}$, which is actually well-conditioned. That puts spurious bright spots on the `prob4` heat map.

Sorting both lists first (`np.sort_complex`) fixes it:

| Matrix entries $(x, y)$ | $(50, 50)$ | $(-80, 40)$ | $(1, 100)$ | $(100, 0.01)$ | $(0, 100)$ |
|---|---|---|---|---|---|
| Relative condition number, sorted | 0.67 | 0.53 | 12 | 168 | $5\times10^{7}$ |

After the fix the expected picture appears: matrices close to normal are well-conditioned, and very non-normal ones are not. Fixing this bug, and the subtle reasoning behind it, is a good exercise.

## Ideas to extend it

- Add a third least-squares solver based on the SVD (`np.linalg.lstsq`) and compare all three.
- Rewrite $I(n)$ with backward recursion, $I(n-1) = (1 - I(n))/n$, which is stable, and plot the error again.
- Show how ridge regularization improves $\kappa(A^\mathsf{T}A + \lambda I)$ as $\lambda$ grows.

## Further reading

- [Condition number (Wikipedia)](https://en.wikipedia.org/wiki/Condition_number)
- [Wilkinson's polynomial](https://en.wikipedia.org/wiki/Wilkinson%27s_polynomial)
- [Catastrophic cancellation](https://en.wikipedia.org/wiki/Catastrophic_cancellation)
- Trefethen & Bau, *Numerical Linear Algebra* (SIAM, 1997)
