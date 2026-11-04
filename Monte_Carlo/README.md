# Monte Carlo Integration: Computing with Randomness

> Estimate π, measure the volume of a 10-dimensional sphere, and integrate functions that defeat classical methods, all by throwing random darts.

| | |
|---|---|
| **Concepts** | Random sampling, numerical integration, the law of large numbers, convergence rates, the curse of dimensionality |
| **Stack** | NumPy, SciPy, Matplotlib |
| **Code** | [`monte_carlo_integration`](monte_carlo_integration) (a Python script without a `.py` extension) |

---

## Why it's cool

To estimate the area of a circle, throw darts at random into the square around it and count how many land inside. That's Monte Carlo, and it hides a remarkable fact.

**Its error shrinks like $1/\sqrt{N}$, no matter how many dimensions there are.** A grid-based method needs $10^{d}$ points to get 10 points per axis in $d$ dimensions, which is impossible beyond a handful of dimensions. Monte Carlo doesn't care. That is why it's the standard tool for high-dimensional problems in finance, physics, and machine learning.

## Where it's used in the real world

- **Finance:** pricing options and derivatives, Value-at-Risk, and portfolio stress tests.
- **Physics and engineering:** radiation transport and particle simulations. The method was named during the Manhattan Project.
- **Computer graphics:** path tracing in film and game rendering (Pixar, Unreal Engine).
- **Machine learning and statistics:** Bayesian inference (MCMC), dropout uncertainty, and reinforcement-learning rollouts.
- **Operations and risk:** project-schedule risk, reliability analysis, and supply-chain simulation.

## The concept in 60 seconds

For a region $\Omega$ with volume $V$, sample $N$ points $x_i$ uniformly from $\Omega$:

$$\int_\Omega f(x)\,dx \;\approx\; \frac{V}{N}\sum_{i=1}^{N} f(x_i)$$

- **Why it works:** by the law of large numbers, the sample average converges to the true average of $f$.
- **How fast:** by the central limit theorem, the error is about $\sigma/\sqrt{N}$. Getting 10× more accuracy costs 100× more samples, but that rate doesn't get worse in higher dimensions.
- **Unit-ball volume:** sample in the cube $[-1,1]^n$ (volume $2^n$). The fraction of points with $\lVert x\rVert < 1$, times $2^n$, estimates the ball's volume.

## What's implemented

| Function | What it does |
|---|---|
| `ball_volume(n, N)` | Estimates the volume of the $n$-dimensional unit ball with vectorized sampling and norms |
| `mc_integrate1d(f, a, b, N)` | 1-D Monte Carlo integration on $[a, b]$ |
| `mc_integrate(f, mins, maxs, N)` | $n$-D integration over any box, mapping samples from the unit cube onto $[\text{mins}, \text{maxs}]$ |
| `prob4()` | Integrates a **4-D standard normal density** over a box, compares with SciPy's exact multivariate-normal CDF, and plots relative error against $N$ (20 log-spaced values from $10^1$ to $10^5$) next to the theoretical $1/\sqrt{N}$ line |

## Results

Unit-ball volume with $N = 10^6$ samples:

| Dimension $n$ | Monte Carlo estimate | Exact $\pi^{n/2}/\Gamma(n/2+1)$ |
|---|---|---|
| 2 (circle, i.e. π) | 3.1417 | 3.1416 |
| 3 | 4.1907 | 4.1888 |
| 4 | 4.9367 | 4.9348 |
| 6 | 5.1698 | 5.1677 |
| 10 | 2.5938 | 2.5502 |

- **Surprise for learners:** the unit ball's volume peaks around 5 dimensions and then **shrinks toward zero**. In 10-D, only about 0.25% of the surrounding cube lies inside the ball. Almost every dart lands in a corner, so the estimate gets noisier.
- **2-D integral:** $\int_1^3\int_{-2}^1 (3x - 4y + y^2)\,dy\,dx$ estimates to **53.87** with $10^5$ samples. The exact value is 54.

## Run it

```bash
cd Monte_Carlo
python -i monte_carlo_integration
```

```python
>>> ball_volume(2, 10**6)        # ≈ π
>>> mc_integrate(lambda x: x[0]**2 + x[1]**2, [0, 0], [1, 1], 10**5)   # exact: 2/3
>>> prob4()                      # convergence plot
```

## Skills demonstrated

- Vectorized NumPy: sampling, broadcasting, and norms over arrays of millions of points.
- Statistical reasoning about error, and checking an empirical convergence rate against theory.
- Checking results against exact formulas and library references.

## Ideas to extend it

- **Variance reduction:** importance sampling, antithetic variates, or stratified sampling. How much does each improve accuracy at the same $N$?
- **Quasi-Monte Carlo:** use low-discrepancy (Sobol) sequences from `scipy.stats.qmc` for close to $1/N$ convergence.
- Price a European call option and compare with the Black–Scholes formula.

## Further reading

- [Monte Carlo integration (Wikipedia)](https://en.wikipedia.org/wiki/Monte_Carlo_integration)
- [Volume of an n-ball](https://en.wikipedia.org/wiki/Volume_of_an_n-ball)
- [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
