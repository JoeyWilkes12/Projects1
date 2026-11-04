# Three Ways to Take a Derivative: Symbolic, Numerical & Automatic

> Computers differentiate in three fundamentally different ways. This project builds and benchmarks all three, then uses one to clock the speed of an aircraft from two radar dishes.

| | |
|---|---|
| **Concepts** | Finite differences, truncation vs. round-off error, symbolic algebra, automatic differentiation, Jacobians, Chebyshev polynomials |
| **Stack** | SymPy, NumPy, Autograd, Matplotlib |
| **Code** | [`differentiation.py`](differentiation.py) |

---

## Why it's cool

Every neural network you've heard of is trained by computing derivatives, millions of them a second. The method that makes this possible, **automatic differentiation**, is neither the calculus you did by hand (symbolic) nor the "rise over run" approximation (finite differences). It's a third approach that is both exact and fast.

This project compares all three head-to-head. Along the way it shows a counter-intuitive result: **making the step size smaller can make a numerical derivative *less* accurate.**

## Where it's used in the real world

- **Deep learning:** PyTorch, TensorFlow, and JAX are automatic-differentiation engines at heart. Backpropagation is reverse-mode autodiff.
- **Scientific computing:** finite differences drive weather models, fluid dynamics, and heat-transfer simulations.
- **Computer algebra systems:** Mathematica, Maple, and SymPy produce exact derivatives for analysis and code generation.
- **Tracking and navigation:** estimating velocity from position data, as with radar, GPS, and robot sensors.
- **Optimization:** gradients and Jacobians drive Newton's method (see [`../Newtons_method`](../Newtons_method)), gradient descent, and sensitivity analysis.

## The concept in 60 seconds

| Method | How it works | Accuracy | Cost |
|---|---|---|---|
| **Symbolic** (SymPy) | Applies calculus rules to the formula itself | Exact | Slow, and expressions can blow up in size |
| **Finite differences** | $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$ | Approximate, limited by step size $h$ | Cheap; needs only values of $f$ |
| **Automatic** (Autograd) | Applies the chain rule to every operation as the program runs | Exact to machine precision | Fast |

**The step-size trade-off.** A difference quotient has two sources of error:
- **Truncation error**, which shrinks as $h \to 0$. It goes as $O(h)$, $O(h^2)$, or $O(h^4)$, depending on the formula.
- **Round-off error**, which *grows* like $\varepsilon_{\text{machine}}/h$ as $h \to 0$, because you subtract two nearly equal numbers.

So there's a best $h$, and going smaller makes things worse.

## What's implemented

| Function | What it does |
|---|---|
| `prob1()` | The exact derivative of $f(x) = (\sin x + 1)^{\sin(\cos x)}$ via SymPy, converted to a fast NumPy function |
| `fdq1`, `fdq2`, `bdq1`, `bdq2`, `cdq2`, `cdq4` | Forward, backward, and centered difference quotients of order 1, 2, and 4 |
| `prob3(x0)` | Log-log plot of absolute error against $h$ for every quotient, showing the trade-off above |
| `prob4()` | **Radar tracking:** two stations 500 m apart record angles to a plane each second. The code converts the angles to $(x, y)$ positions, then uses forward, centered, and backward differences to estimate the plane's **speed**. |
| `jacobian_cdq2(f, x)` | A finite-difference **Jacobian** for $f:\mathbb{R}^n\to\mathbb{R}^m$, checked against SymPy's exact Jacobian |
| `cheb_poly(x, n)`, `prob6()` | Chebyshev polynomials from their recurrence relation, differentiated with Autograd's `elementwise_grad` |
| `prob7(N)` | **Benchmark:** SymPy against `cdq4` against Autograd on N random points, plotting computation time against error |

## Results

Absolute error in $f'(1)$ for each quotient as the step size $h$ shrinks:

| Method | Order | $h = 10^{-1}$ | $h = 10^{-3}$ | $h = 10^{-5}$ | $h = 10^{-8}$ |
|---|---|---|---|---|---|
| Forward `fdq1` | 1 | 7.2e-02 | 7.7e-04 | 7.7e-06 | 9.1e-09 |
| Forward `fdq2` | 2 | 1.1e-02 | 8.6e-07 | 1.1e-10 | 4.2e-08 ↑ |
| Centered `cdq2` | 2 | 4.3e-03 | 4.3e-07 | **3.8e-11** | 9.1e-09 ↑ |
| Centered `cdq4` | 4 | 8.4e-05 | **9.4e-13** | 6.7e-12 ↑ | 1.1e-08 ↑ |
| **Autograd** | exact | | | | **0.0** |

- **↑** marks where shrinking $h$ made the answer *worse*. That's round-off taking over.
- The 4th-order formula is most accurate at a relatively *large* $h = 10^{-3}$.
- Autograd matches SymPy's exact derivative to machine precision.

**Jacobian check:** for $f(x,y) = (x + y,\ xy^2)$ at $(2, 3)$, `jacobian_cdq2` returns $\begin{bmatrix}1 & 1\\ 9 & 12\end{bmatrix}$, matching SymPy exactly.

## Run it

```bash
cd Differentiation
python -i differentiation.py
```

```python
>>> prob3(1.0)        # error vs. step size plot
>>> prob7(200)        # time vs. accuracy benchmark
>>> test_jacobian()
```

- **Data:** `prob4` (radar tracking) needs `plane.npy`, which isn't included in the repository.
- **Compatibility:** a few lines pass `1e3` as a sample count to `linspace`, which current NumPy rejects. Use `int(1e3)` instead.

## Skills demonstrated

- Numerical analysis: deriving and implementing formulas of several orders, and analyzing error empirically.
- Fluency with three paradigms (symbolic, numeric, and autodiff) and knowing when to use each.
- Turning raw sensor data into a physical estimate (aircraft speed).
- Benchmarking and visualizing performance trade-offs.

## Ideas to extend it

- Add **complex-step differentiation**, $f'(x) \approx \mathrm{Im} f(x + ih)/h$. It has no subtraction, so no round-off blow-up, and $h = 10^{-20}$ works.
- Re-implement the benchmark in JAX and compare forward-mode with reverse-mode autodiff.
- Use Richardson extrapolation to build higher-order quotients automatically.

## Further reading

- [Numerical differentiation (Wikipedia)](https://en.wikipedia.org/wiki/Numerical_differentiation)
- [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR, 2018)
