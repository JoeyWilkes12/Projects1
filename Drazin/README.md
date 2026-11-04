# Drazin Inverse, Effective Resistance & Link Prediction

> Who should you befriend next? This project answers that by treating a social network as an **electrical circuit**, using a generalized matrix inverse that works even when the ordinary inverse doesn't exist.

| | |
|---|---|
| **Concepts** | Singular matrices, matrix index, Drazin inverse, Schur decomposition, graph Laplacian, effective resistance |
| **Stack** | NumPy, SciPy |
| **Code** | [`Drazin_Inverse.py`](Drazin_Inverse.py) |

---

## Why it's cool

Many important matrices are **singular**: they have no ordinary inverse. The graph Laplacian is one of them. The **Drazin inverse** is a generalized inverse that behaves like a true inverse on the "invertible part" of a matrix and is zero on the rest.

With it you can compute **effective resistance**. Imagine every friendship is a 1-ohm resistor and ask how easily current flows between two people. Many short, parallel paths mean low resistance, which means a strong hidden connection. **Two people with low resistance who aren't yet connected are the best candidates for a new link.** That is a principled "People You May Know".

## Where it's used in the real world

- **Recommendations:** friend, follower, and connection suggestions on social platforms.
- **Network science:** commute time and random-walk distance in graphs, and community detection.
- **Electrical engineering:** analyzing resistor networks and power grids.
- **Markov chains:** the Drazin inverse gives mean first-passage times and long-run behavior of chains.
- **Systems of differential equations:** solving singular systems $A\mathbf{x}' = B\mathbf{x}$ in control theory.
- **Chemistry:** the "Kirchhoff index" of molecular graphs.

## The concept in 60 seconds

**Index.** The index of a square matrix $A$ is the smallest $k$ where $\text{rank}(A^k) = \text{rank}(A^{k+1})$. An invertible matrix has index 0.

**Drazin inverse.** $A^D$ is the unique matrix satisfying:

$$AA^D = A^DA, \qquad A^{k+1}A^D = A^k, \qquad A^DAA^D = A^D$$

If $A$ is invertible, then $A^D = A^{-1}$.

**How to compute it.** Use the Schur decomposition to split $A$'s space into the part with nonzero eigenvalues (the invertible piece) and the part with zero eigenvalues (the nilpotent piece). Invert only the first part.

**Effective resistance.**
1. Build the graph Laplacian $L = D - A$, where $D$ holds each node's degree.
2. For each node $j$, replace row $j$ of $L$ with the identity row. This "grounds" node $j$.
3. Take the Drazin inverse of the result. Its diagonal gives the resistance from every node to $j$.

## What's implemented

| Function / class | What it does |
|---|---|
| `index(A)` | Computes the matrix index from successive matrix powers and their ranks |
| `is_drazin(A, Ad, k)` | Checks all three defining properties of the Drazin inverse |
| `drazin_inverse(A)` | Computes $A^D$ with two **sorted Schur decompositions** (SciPy `la.schur(sort=...)`) |
| `effective_resistance(A)` | The full $n \times n$ effective-resistance matrix of an undirected graph, using the Laplacian and grounding trick above |
| `LinkPredictor(filename)` | Reads a social network from CSV, builds its adjacency matrix, and precomputes effective resistance |
| `LinkPredictor.predict_link(node=None)` | Returns the unconnected pair with the lowest resistance, either across the whole network or for one person |
| `LinkPredictor.add_link(a, b)` | Adds an edge and recomputes resistances, so predictions change as the network grows |

## Results

- Both hand-built test cases pass `is_drazin`: an index-1 upper-triangular matrix and an index-3 nilpotent matrix, whose Drazin inverse is all zeros.
- `drazin_inverse` produces a matrix that passes all three checks.
- On a 3-node path A–B–C, effective resistance matches circuit theory exactly. Two 1-ohm resistors in series give 2 ohms.

```
     A  B  C
A [  0  1  2 ]
B [  1  0  1 ]
C [  2  1  0 ]
```

## Run it

```bash
cd Drazin
python -i Drazin_Inverse.py
```

```python
>>> import numpy as np
>>> effective_resistance(np.array([[0,1,0],[1,0,1],[0,1,0]]))
>>> test_is_drazin()
```

`LinkPredictor` needs `social_network.csv` in this folder. That file isn't included in the repository; each line is one friendship, written as `name1,name2`.

## Skills demonstrated

- Advanced numerical linear algebra: matrix decompositions, generalized inverses, and working with rank deficiency.
- Connecting abstract math (circuit theory) to a product feature (connection recommendations).
- Clear object-oriented design: a stateful predictor that updates as links are added.

## Ideas to extend it

- Compute all resistances at once with the Moore–Penrose pseudo-inverse of $L$ and compare the speed.
- Evaluate prediction quality: hide 10% of edges and measure how many the model recovers.
- Compare against simpler baselines such as common-neighbor counts and the Jaccard index.

## Further reading

- [Drazin inverse (Wikipedia)](https://en.wikipedia.org/wiki/Drazin_inverse)
- [Resistance distance](https://en.wikipedia.org/wiki/Resistance_distance)
- Liben-Nowell & Kleinberg, *The Link Prediction Problem for Social Networks* (2007)
