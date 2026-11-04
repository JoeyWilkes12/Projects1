# Newton's Method: Root Finding, Retirement Math & Fractals

> A 300-year-old algorithm that doubles its correct digits every step, used here to solve a retirement-savings problem, tame a chemical system, and draw fractals in the complex plane.

| | |
|---|---|
| **Concepts** | Root finding, quadratic convergence, Jacobians, backtracking (damped Newton), basins of attraction |
| **Stack** | NumPy, Autograd, Matplotlib |
| **Code** | [`newtons_method.py`](newtons_method.py) |

---

## Why it's cool

To solve $f(x) = 0$, Newton's method follows the tangent line down to the x-axis and repeats. Near a root it converges **quadratically**: 1 correct digit, then 2, 4, 8, and so on. That speed is why it sits inside nearly every scientific computing library.

But it can also fail in spectacular ways:
- On $f(x) = \sqrt[3]{x}$ it moves *away* from the root, and every step lands twice as far out.
- Colour each starting point in the complex plane by the root it eventually reaches and you get a **fractal**.

This project explores the speed, the failures, and the fixes.

## Where it's used in the real world

- **Optimization and machine learning:** Newton and quasi-Newton methods (L-BFGS) train logistic regression and other models. Setting a gradient to zero is a root-finding problem.
- **Finance:** solving for yield to maturity, the internal rate of return (IRR), and option implied volatility.
- **Engineering simulation:** nonlinear finite-element solvers, circuit simulators (SPICE), and power-flow analysis on electrical grids.
- **Computer graphics:** ray–surface intersection, and inverse kinematics for animation and robotics.
- **Hardware:** fast reciprocal and square-root routines in CPUs and GPUs.

## The concept in 60 seconds

**Scalar update:**

$$x_{k+1} = x_k - \alpha\,\frac{f(x_k)}{f'(x_k)}$$

**Vector update**, for $f:\mathbb{R}^n \to \mathbb{R}^n$: solve $Df(\mathbf{x}_k)\,\mathbf{y}_k = f(\mathbf{x}_k)$, then set $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha\,\mathbf{y}_k$. Here $Df$ is the Jacobian matrix. The code solves the linear system rather than inverting the Jacobian, which is faster and more stable.

**Backtracking ($\alpha$).** With $\alpha = 1$, Newton can overshoot. Shrinking the step can turn divergence into convergence, and can even change which root you land on.

**Basins of attraction.** For $f(z) = z^3 - 1$ there are three complex roots. Start Newton from each point on a grid, colour the point by the root it reaches, and the boundaries between colours turn out to be fractal.

## What's implemented

| Function | What it does |
|---|---|
| `newton(f, x0, Df, tol, maxiter, alpha)` | Newton's method for scalars and vectors; returns `(root, converged, iterations)` |
| `prob2(N1, N2, P1, P2)` | **Retirement planning:** finds the interest rate $r$ at which depositing $P_1$ a year for $N_1$ years funds withdrawals of $P_2$ a year for $N_2$ years |
| `optimal_alpha(f, x0, Df)` | Sweeps 10,000 values of $\alpha$ in $(0, 1]$ and plots iterations to converge; returns the best $\alpha$ |
| `prob6()` | **Bioremediation model:** a 2-D nonlinear system. Searches for a starting point that converges to different equilibria depending on $\alpha$. |
| `plot_basins(f, Df, zeros, domain, res)` | Runs Newton on a whole complex grid at once (vectorized) and draws the basins of attraction |
| `test_newton()` | Assertion tests: $x^4 - 3$, $e^x - 2$, the cube-root failure case, a 2-D system, and a zero-derivative guard (`ZeroDivisionError`) |

## Results

- **Retirement:** depositing \$2,000 a year for 30 years, then withdrawing \$8,000 a year for 20 years, requires an annual return of **$r$ = 3.878%**.
- **Cube root, a classic failure:** for $f(x) = \sqrt[3]{x}$ from $x_0 = 0.01$:

  | $\alpha$ | Result after 15 steps |
  |---|---|
  | 1.0 | **Diverges** to $x \approx -327.7$; each step multiplies $x$ by −2 |
  | 0.4 | **Converges** to $6.4\times10^{-7}$ |

- **Optimal step size:** the $\alpha$ sweep finds **α ≈ 0.3343**. The theory explains why. For the cube root, $x - \alpha f/f' = x(1 - 3\alpha)$, so $\alpha = 1/3$ lands exactly on the root in one step.
- **Bioremediation:** found the starting point $(-0.25,\ 0.2123)$, which converges to $(0, \pm1)$ with $\alpha = 1$ but to $(3.75, 0.25)$ with $\alpha = 0.55$.
- **Fractals:** `basins_test()` renders the three-colour Newton fractal for $z^3 - 1$, and the basins of $z^3 - z$.

## Run it

```bash
cd Newtons_method
python -i newtons_method.py
```

Loading the script also runs `test_optimal_alpha()`, which opens a plot.

```python
>>> test_newton()
>>> prob2(30, 20, 2000, 8000)   # ≈ 0.03878
>>> basins_test()               # fractal plots
```

No external data is needed.

## Skills demonstrated

- Implementing a numerical algorithm that works for both scalar and vector inputs.
- Diagnosing convergence failures and fixing them in a principled way, backed by theory.
- Applying math to finance and to chemical-systems modeling.
- Vectorized computation over a million-point complex grid, and scientific visualization.

## Ideas to extend it

- Add a line search (Armijo condition) so $\alpha$ adapts automatically at each step.
- Implement the secant method and Broyden's method, which need no derivatives, and compare convergence rates.
- Colour the basin plots by iteration count as well as by root, for more detailed fractal images.

## Further reading

- [Newton's method (Wikipedia)](https://en.wikipedia.org/wiki/Newton%27s_method)
- [Newton fractal](https://en.wikipedia.org/wiki/Newton_fractal)
- [Rate of convergence](https://en.wikipedia.org/wiki/Rate_of_convergence)
