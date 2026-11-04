# PageRank: Ranking Anything by Who Points to It

> The algorithm that launched Google, implemented three different ways and used to rank web pages, college sports teams, and Hollywood actors.

| | |
|---|---|
| **Concepts** | Directed graphs, Markov chains, eigenvectors, linear systems, power iteration, damping |
| **Stack** | NumPy, NetworkX |
| **Code** | [`PageRank.py`](PageRank.py) |

---

## Why it's cool

PageRank turns a vague question, "what's important?", into linear algebra. The idea: **a node is important if important nodes point to it.** That sounds circular, but it has an exact answer. It is the dominant eigenvector of a matrix built from the graph.

The same few lines of math work on any network, which is why this project ranks three completely different things with one class.

## Where it's used in the real world

- **Web search:** the original Google ranking signal for web pages.
- **Sports analytics:** ranking teams by whom they beat, not just how often they win.
- **Science:** ranking papers and journals by citations (Eigenfactor).
- **Biology:** finding key proteins in interaction networks (GeneRank).
- **Social media:** "who to follow" and influencer detection.
- **Infrastructure:** finding critical nodes in road and power networks, and spotting fraud rings in transaction graphs.

## The concept in 60 seconds

Picture a **random surfer** who clicks links forever. A page's PageRank is the long-run fraction of time the surfer spends on it.

1. **Adjacency matrix.** $A_{ij}$ is the weight of the link from page $j$ to page $i$. Each column is normalized to sum to 1, giving $\hat{A}$: from page $j$, the surfer picks an outgoing link at random.
2. **Sinks.** A page with no outgoing links would trap the surfer. Its column is replaced with $1/n$ everywhere, so the surfer jumps to a random page.
3. **Damping.** With probability $\varepsilon$ (usually 0.85) the surfer follows a link; otherwise they teleport anywhere. This guarantees a unique answer.

The PageRank vector $\mathbf{p}$ satisfies:

$$\mathbf{p} = \varepsilon \hat{A}\mathbf{p} + \frac{1-\varepsilon}{n}\mathbf{1}$$

This is a **Markov chain** steady state, and it can be solved three equivalent ways.

## What's implemented

| Component | What it does |
|---|---|
| `DiGraph(A, labels)` | Validates the labels, removes sinks, and column-normalizes the adjacency matrix |
| `DiGraph.linsolve()` | Solves $(I - \varepsilon\hat{A})\mathbf{p} = c\mathbf{1}$ directly as a **linear system** |
| `DiGraph.eigensolve()` | Finds the eigenvector of $B = \varepsilon\hat{A} + \frac{1-\varepsilon}{n}E$ for its largest eigenvalue |
| `DiGraph.itersolve()` | **Power iteration**: repeatedly applies the update until the change falls below a tolerance. This is the approach that scales to billions of pages. |
| `get_ranks(d)` | Sorts the labels by PageRank score |
| `rank_websites()` | Ranks pages in the **Stanford web graph** dataset |
| `rank_ncaa_teams()` | Ranks **NCAA teams**: each loss is a "vote" from the loser to the winner, so beating strong teams counts for more |
| `rank_actors()` | Ranks actors from the **IMDb Top 250** with NetworkX. Lower-billed actors "vote" for higher-billed co-stars. |

## Results

On a 4-node test graph, all three from-scratch solvers agree with each other and with NetworkX's reference `nx.pagerank` to four decimal places:

| Node | `linsolve` | `eigensolve` | `itersolve` | `networkx.pagerank` |
|---|---|---|---|---|
| a | 0.0958 | 0.0958 | 0.0958 | 0.0958 |
| b | 0.2742 | 0.2742 | 0.2742 | 0.2742 |
| c | **0.3559** | **0.3559** | **0.3559** | **0.3559** |
| d | 0.2742 | 0.2742 | 0.2742 | 0.2742 |

Ranking: `c > b = d > a`.

**Why three solvers?** They trade off differently:
- The linear solve is exact, but costs $O(n^3)$.
- The eigen-decomposition shows the theory directly.
- Power iteration needs only matrix-vector products. That's the only option at web scale, where $\hat{A}$ is enormous and sparse.

## Run it

```bash
cd PageRank
python
```

```python
>>> import numpy as np
>>> from PageRank import DiGraph, get_ranks
>>> A = np.array([[0,0,0,0],[1,0,1,0],[1,0,0,1],[1,0,1,0]])
>>> get_ranks(DiGraph(A, labels=list("abcd")).itersolve())
```

The ranking functions need `web_stanford.txt`, an NCAA results CSV, and `top250movies.txt`. These aren't included in the repository.

## Skills demonstrated

- Turning a ranking problem into an eigenvalue problem, and solving it three ways with an understanding of each one's cost.
- Checking custom implementations against a standard library (NetworkX).
- Building networks from raw data: edge lists, game results, and film casts.

## Ideas to extend it

- Use `scipy.sparse` so power iteration scales to millions of nodes.
- Plot how the rankings change as the damping factor $\varepsilon$ varies.
- Add personalized PageRank (teleporting to a chosen set of nodes) for recommendations.

## Further reading

- [PageRank (Wikipedia)](https://en.wikipedia.org/wiki/PageRank)
- Brin & Page, *The Anatomy of a Large-Scale Hypertextual Web Search Engine* (1998)
- [Power iteration](https://en.wikipedia.org/wiki/Power_iteration)
