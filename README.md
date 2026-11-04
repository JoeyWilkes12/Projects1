# Projects1 — Applied Algorithms & Computational Mathematics in Python

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-graphs-orange)
![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)

A monorepo of self-contained Python projects that implement core algorithms **from first principles** and then apply them to real problems — ranking web pages, predicting social-network connections, classifying handwritten digits, tracking an aircraft from radar angles, and solving Wordle.

Each project pairs a from-scratch implementation with validation: against a reference library (NetworkX, SciPy, SymPy, Autograd), against known analytical results, or with `pytest` test suites.

**Every project folder has its own README** written for two audiences: reviewers assessing the engineering, and learners meeting the concept for the first time. Each one covers why the idea is interesting, where it's used in industry, a 60-second concept primer, what's implemented, results from actually running the code, and ideas for extending it.

---

## Contents

- [Projects at a glance](#projects-at-a-glance)
- [Project details](#project-details)
  - [Graphs & networks](#graphs--networks)
  - [Data structures & search](#data-structures--search)
  - [Numerical analysis](#numerical-analysis)
  - [Probability & machine learning](#probability--machine-learning)
  - [Software engineering](#software-engineering)
  - [Applied project: Wordle solver](#applied-project-wordle-solver)
- [Skills demonstrated](#skills-demonstrated)
- [Getting started](#getting-started)
- [Repository layout](#repository-layout)
- [Notes & known limitations](#notes--known-limitations)

---

## Projects at a glance

| Project | Domain | What it does | Core tools |
|---|---|---|---|
| [**BFS**](BFS/) | Graph theory | Adjacency-dictionary graph class with BFS traversal and shortest paths; "Six Degrees of Kevin Bacon" analysis on IMDb data | Python, NetworkX |
| [**PageRank**](PageRank/) | Graph theory / linear algebra | PageRank via linear solve, eigenvector, and power iteration; ranks web pages, NCAA teams, and actors | NumPy, SciPy, NetworkX |
| [**Drazin**](Drazin/) | Linear algebra / networks | Drazin inverse and effective resistance; link prediction ("people you may know") for a social network | NumPy, SciPy |
| [**Binary_Trees**](Binary_Trees/) | Data structures | Singly linked list, binary search tree, and self-balancing AVL tree; empirical build/search benchmarks | NumPy, Matplotlib, NetworkX |
| [**KNN**](KNN/) | Data structures / ML | k-d tree with nearest-neighbor search and a k-nearest-neighbors classifier applied to MNIST digits | NumPy, SciPy |
| [**Markov_Chains**](Markov_Chains/) | Probability | Random transition matrices, weather forecasting, steady-state distributions, and a Markov text generator | NumPy, SciPy, pytest |
| [**Monte_Carlo**](Monte_Carlo/) | Numerical integration | Monte Carlo estimates of n-ball volumes and multi-dimensional integrals, with error-convergence analysis | NumPy, SciPy |
| [**Newtons_method**](Newtons_method/) | Root finding | Newton's method in ℝⁿ with backtracking, step-size tuning, and complex-plane basins of attraction | NumPy, Autograd |
| [**Differentiation**](Differentiation/) | Numerical calculus | Symbolic, finite-difference, and automatic differentiation compared for accuracy and speed | SymPy, NumPy, Autograd |
| [**Conditional_Stability**](Conditional_Stability/) | Numerical stability | Condition numbers, Wilkinson-polynomial perturbation, and stable vs. unstable least squares | NumPy, SciPy, SymPy |
| [**Unit_testing**](Unit_testing/) | Software engineering | Test-driven verification of utility functions, a `Fraction` class, and a SET card-game counter | pytest |
| [**Wordle**](Wordle/) | Applied / NLP | Constraint-based Wordle solver backed by curated 5-letter-word datasets | pandas, NumPy, Matplotlib |

> New to graph theory? Several projects build on it — [Wikipedia's overview](https://en.wikipedia.org/wiki/Graph_theory) is a good primer.

---

## Project details

> Short summaries below. Follow each **Read more** link for the full write-up.

### Graphs & networks

#### BFS — Breadth-First Search &nbsp;·&nbsp; [`BFS/breadth_first_search.py`](BFS/breadth_first_search.py)
- Implements a `Graph` class backed by an adjacency dictionary, with node/edge insertion and removal, **breadth-first traversal**, and **shortest-path** search.
- Applies the same ideas at scale with NetworkX in `MovieGraph`, a bipartite actor–movie graph built from IMDb data, to compute an actor's "Bacon number" (degrees of separation) and plot how far every actor in the network is from a chosen actor, along with the average.

**[Read more →](BFS/README.md)**

#### PageRank &nbsp;·&nbsp; [`PageRank/PageRank.py`](PageRank/PageRank.py)
- A `DiGraph` class that removes sink nodes, normalizes the adjacency matrix, and computes PageRank **three ways**: a direct linear-system solve, the dominant eigenvector, and power iteration with a convergence tolerance.
- Applied to three ranking problems:
  - **Web pages:** the Stanford web graph.
  - **Sports:** NCAA teams, with each game result treated as a weighted edge.
  - **Film:** actors from the IMDb Top 250, via NetworkX.

**[Read more →](PageRank/README.md)**

#### Drazin — Drazin Inverse & Link Prediction &nbsp;·&nbsp; [`Drazin/Drazin_Inverse.py`](Drazin/Drazin_Inverse.py)
- Computes the matrix **index** and **Drazin inverse** (a generalized pseudo-inverse) with a verification routine.
- Uses the graph Laplacian to compute **effective resistance** between every pair of nodes. Lower resistance means a stronger implied connection.
- `LinkPredictor` reads a social network from CSV and recommends the most likely new connection, either globally or for a given person. This is the same idea that drives "friend suggestion" features.

**[Read more →](Drazin/README.md)**

### Data structures & search

#### Binary_Trees &nbsp;·&nbsp; [`Binary_Trees/binary_trees.py`](Binary_Trees/binary_trees.py)
- From-scratch **singly linked list** (iterative and recursive search), **binary search tree** (insert, find, remove with all child cases), and **AVL tree** (all four rotation cases for self-balancing).
- Benchmarks build and search times across the three structures on growing inputs and plots them on log scales, showing the gap between O(n) and O(log n) behavior in practice.

**[Read more →](Binary_Trees/README.md)**

#### KNN — k-d Trees & Nearest Neighbors &nbsp;·&nbsp; [`KNN/k_nearest_neighbor.py`](KNN/k_nearest_neighbor.py)
- Exhaustive (brute-force) nearest-neighbor search as a baseline.
- A custom **k-d tree** with insertion and a pruned recursive nearest-neighbor query.
- A `KNeighborsClassifier` (`fit`/`predict`, majority vote over `scipy.spatial.KDTree` neighbors) evaluated on a subset of the **MNIST** handwritten-digit dataset, with classification accuracy reported.

**[Read more →](KNN/README.md)**

### Numerical analysis

#### Conditional_Stability — Conditioning & Stability &nbsp;·&nbsp; [`Conditional_Stability/conditioning-stability.py`](Conditional_Stability/conditioning-stability.py)
- Matrix condition numbers from singular values, tested on orthonormal (κ = 1) and singular (κ = ∞) matrices.
- Perturbs the coefficients of the **Wilkinson polynomial** and plots how far its roots move, showing how sensitive root-finding can be.
- Computes and visualizes eigenvalue condition numbers over a 2-D parameter grid.
- Compares least squares solved with the **normal equations** against the **QR decomposition** to show why the QR approach is preferred.
- Shows how an analytically correct formula can still be numerically unstable, with errors plotted on a log scale.

**[Read more →](Conditional_Stability/README.md)**

#### Differentiation &nbsp;·&nbsp; [`Differentiation/differentiation.py`](Differentiation/differentiation.py)
- First- and second-order forward/backward and second- and fourth-order **centered difference quotients**, with log-log error analysis against step size *h*.
- A finite-difference **Jacobian** and **Chebyshev polynomial** derivatives via Autograd.
- **Applied problem:** estimates the speed of an aircraft tracked by two radar stations, given only their angle readings.
- **Benchmark:** compares symbolic (SymPy), numerical (difference quotients), and automatic (Autograd) differentiation on computation time against accuracy.

**[Read more →](Differentiation/README.md)**

#### Newtons_method &nbsp;·&nbsp; [`Newtons_method/newtons_method.py`](Newtons_method/newtons_method.py)
- A general Newton's method for scalar and vector-valued functions, with convergence reporting and a backtracking parameter α.
- **Applied problems:** solving for the interest rate in a retirement savings/withdrawal model, and finding initial points for a nonlinear bioremediation system.
- Searches for the step size α that converges in the fewest iterations.
- Plots **basins of attraction** in the complex plane (Newton fractals).

**[Read more →](Newtons_method/README.md)**

#### Monte_Carlo — Monte Carlo Integration &nbsp;·&nbsp; [`Monte_Carlo/monte_carlo_integration`](Monte_Carlo/monte_carlo_integration)
- Estimates the volume of the unit ball in *n* dimensions by random sampling.
- 1-D and general *n*-D Monte Carlo integration over box domains.
- Checks a 4-D Gaussian integral against SciPy's multivariate-normal CDF and confirms the expected **O(1/√N)** error convergence on a log-log plot.

**[Read more →](Monte_Carlo/README.md)**

### Probability & machine learning

#### Markov_Chains &nbsp;·&nbsp; [`Markov_Chains/markov_chains.py`](Markov_Chains/markov_chains.py)
- Generates random column-stochastic transition matrices and simulates two-state and four-state weather forecasts.
- Computes **steady-state distributions** by iteration, raising an error if the iteration does not converge.
- `SentenceGenerator` builds a word-to-word transition matrix from a text corpus and "babbles" new sentences.
- Covered by a `pytest` suite: [`test_markov.py`](Markov_Chains/test_markov.py).

**[Read more →](Markov_Chains/README.md)**

### Software engineering

#### Unit_testing &nbsp;·&nbsp; [`Unit_testing/`](Unit_testing/)
- [`specs.py`](Unit_testing/specs.py) contains utility functions, a `Fraction` class with arithmetic and comparison operators, and a counter for valid sets in the card game **SET**, including input validation.
- [`test-specs.py`](Unit_testing/test-specs.py) is a `pytest` suite that uses fixtures, exception assertions, and edge cases chosen to expose bugs, such as smallest prime factors of perfect squares and leap-year month lengths.

**[Read more →](Unit_testing/README.md)**

### Applied project: Wordle solver

#### Wordle &nbsp;·&nbsp; [`Wordle/Wordle_Solver_v0.py`](Wordle/Wordle_Solver_v0.py) &nbsp; *(v0, in active development)*
- A constraint-propagation solver. After each guess, it narrows the candidate list using green (right letter, right slot), yellow (right letter, wrong slot), and gray (letter not in word) feedback. It accounts for a repeated letter coming back gray when another copy of it is already confirmed.
- Shows a letter-frequency chart of the remaining candidates to help pick the next guess.
- Ships with three curated, feature-engineered word datasets:

  | Dataset | Words |
  |---|---|
  | NLTK words corpus (5-letter subset) | 8,497 |
  | Community 5-letter word list | 3,624 |
  | Joined NLTK + community list | 9,795 |

  Each row encodes the letter in each position, a one-hot indicator for each letter, and (in the NLTK set) vowel and consonant counts.

**Input notation:** type each letter of your guess. Follow a letter with `!` if it came back green or `?` if it came back yellow. Leave gray letters bare.

```python
from Wordle_Solver_v0 import Wordle_Solver   # run from inside Wordle/

ws = Wordle_Solver(wordle_length=5)
ws.new_guess_results("wea!r!y")      # 'a' and 'r' are green
ws.new_guess_results("apa!r!t")      # narrows the candidates further
ws.new_guess_results("s!h!a!r!d!")   # solved: "shard"
```

**[Read more →](Wordle/README.md)**

---

## Skills demonstrated

- **Algorithms & data structures:** graph traversal, shortest paths, BSTs and AVL rotations, k-d trees, nearest-neighbor search.
- **Numerical linear algebra:** eigenvalue problems, SVD-based conditioning, QR least squares, generalized inverses, graph Laplacians.
- **Numerical methods:** finite differences, automatic differentiation, Newton's method, Monte Carlo integration, convergence and error analysis.
- **Applied modeling:** web and sports ranking, social link prediction, digit classification, radar tracking, financial root-finding, word games.
- **Engineering practice:** object-oriented design, validation against reference libraries, `pytest` suites with fixtures, empirical benchmarking.

---

## Getting started

### Prerequisites
- Python 3
- The scientific Python stack used across the projects:

```bash
pip install -r requirements.txt
```

> `Binary_Trees` imports `graphviz_layout`, which also needs [Graphviz](https://graphviz.org/) and `pygraphviz` installed.

### Run a project
Each project is a standalone module. Run it from inside its own directory so that relative data paths resolve:

```bash
cd PageRank
python PageRank.py
```

### Run the test suites

```bash
cd Unit_testing && python -m pytest test-specs.py
```

```bash
cd Markov_Chains && python -m pytest test_markov.py
```

---

## Repository layout

```
Projects1/
├── BFS/                     # Graph class, BFS, IMDb "Kevin Bacon" network
├── Binary_Trees/            # Linked list, BST, AVL tree + benchmarks
├── Conditional_Stability/   # Condition numbers & numerical stability
├── Differentiation/         # Symbolic vs. numerical vs. automatic differentiation
├── Drazin/                  # Drazin inverse, effective resistance, link prediction
├── KNN/                     # k-d tree & k-nearest-neighbors classifier (MNIST)
├── Markov_Chains/           # Markov chains, steady states, text generation (+ tests)
├── Monte_Carlo/             # Monte Carlo integration
├── Newtons_method/          # Newton's method, backtracking, basins of attraction
├── PageRank/                # PageRank: web, NCAA, and actor rankings
├── Unit_testing/            # pytest-driven development (+ tests)
└── Wordle/                  # Wordle solver + curated 5-letter word datasets
```

---

## Notes & known limitations

- **External datasets are not included.** Several applied functions expect data files in the working directory that aren't in this repository: `movie_data.txt`, `web_stanford.txt`, `top250movies.txt`, `social_network.csv`, `mnist_subset.npz`, `plane.npy`, `stability_data.npy`, `english.txt`, and text corpora for the Markov generator. The core algorithms and the test suites run without them.
- `Monte_Carlo/monte_carlo_integration` is a Python script without a `.py` extension. Run it with `python monte_carlo_integration`.
- The Wordle solver is a **v0**. Planned work includes contradiction checking for mistyped feedback, an automated self-play mode, and tracking how many times each letter appears.
