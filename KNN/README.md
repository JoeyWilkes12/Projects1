# k-d Trees & k-Nearest-Neighbor Classification

> Teach a computer to read handwritten digits with no neural network, using only "you are who your neighbors are" plus a clever tree for finding those neighbors quickly.

| | |
|---|---|
| **Concepts** | Nearest-neighbor search, k-d trees, space partitioning, branch-and-bound pruning, supervised classification |
| **Stack** | NumPy, SciPy |
| **Code** | [`k_nearest_neighbor.py`](k_nearest_neighbor.py) |

---

## Why it's cool

k-nearest neighbors (kNN) is the most intuitive machine-learning model there is. To label a new point, find the $k$ most similar training points and take a majority vote. There is no training step and no gradient descent.

The hard part is speed. Comparing against every one of 60,000 images for every prediction is slow. A **k-d tree** splits space like a game of 20 Questions ("is the 3rd coordinate bigger than 0.4?") and **prunes** whole regions that can't contain a closer point. In low dimensions that typically turns a linear scan into a logarithmic search.

## Where it's used in the real world

- **Recommendations:** "customers who liked this also liked…" (nearest neighbors in taste space).
- **Search by similarity:** reverse image search, music fingerprinting, and the vector databases that power semantic search and AI retrieval (RAG).
- **Maps and GIS:** "find the nearest coffee shop", and geographic range queries.
- **Graphics and games:** collision detection, ray tracing, and point-cloud processing for 3-D scans and LiDAR.
- **Anomaly detection:** a point far from all its neighbors is suspicious (fraud, intrusions).

## The concept in 60 seconds

**The nearest-neighbor problem.** Given training points $X$ and a target $z$, find $\arg\min_{x \in X} \lVert x - z \rVert$.
- **Brute force:** compute every distance, $O(mk)$ per query.

**k-d tree.** A binary tree where each level splits on one coordinate, cycling through them: depth 0 splits on $x_0$, depth 1 on $x_1$, and so on.
- **Insert:** walk down comparing the current pivot coordinate, then attach the new point as a leaf.
- **Query:** walk toward the target, tracking the best distance $d$ found so far. On the way back up, only search the other branch if the splitting plane is **closer than $d$**. If the sphere of radius $d$ around the target doesn't cross the plane, that whole subtree is skipped.

**kNN classification.** Find the $k$ nearest training points and return the most common label among them.

## What's implemented

| Component | What it does |
|---|---|
| `exhaustive_search(X, z)` | A vectorized brute-force baseline using NumPy broadcasting |
| `KDTNode`, `KDT` | A from-scratch k-d tree with type checking and dimension checking |
| `KDT.insert` | Recursive insertion with the pivot coordinate cycling by depth; rejects duplicates |
| `KDT.query` | Recursive nearest-neighbor search with **branch-and-bound pruning** |
| `KDT.__str__` | Prints the tree level by level, showing each node's pivot axis |
| `KNeighborsClassifier` | A scikit-learn-style `fit` / `predict` interface; neighbors come from `scipy.spatial.KDTree`, and the label is decided by majority vote |
| `prob6(n_neighbors)` | Classifies 500 **MNIST** handwritten-digit test images (784-pixel vectors) and reports accuracy |

## Results

- **Correctness:** on 2,000 random points in 5 dimensions, `KDT.query` returned the same nearest neighbor and distance as brute force in **200 out of 200** random queries.
- **MNIST:** `Sixth_problem_Test()` sweeps $k = 1, \dots, 9$ to see how the number of voters changes accuracy. kNN on raw MNIST pixels is commonly reported to reach the mid-90s percent. This repository doesn't include the dataset, so run it yourself to get exact numbers for this subset.

## Run it

```bash
cd KNN
python -i k_nearest_neighbor.py
```

```python
>>> import numpy as np
>>> X = np.random.random((1000, 3)); z = np.random.random(3)
>>> tree = KDT()
>>> for x in X: tree.insert(x)
>>> tree.query(z), exhaustive_search(X, z)   # should match
```

`prob6` needs `mnist_subset.npz` (with keys `X_train`, `y_train`, `X_test`, `y_test`). It isn't included in the repository.

## Skills demonstrated

- Designing a recursive spatial data structure, including the pruning logic that makes it fast.
- Validating a custom implementation against a brute-force reference.
- Applying it to a real computer-vision dataset, with an API modeled on scikit-learn.

## Ideas to extend it

- Benchmark `KDT.query` against brute force as the dimension grows, and watch the **curse of dimensionality** erase the speed-up.
- Use distance-weighted voting, so closer neighbors count more.
- Compare against approximate nearest-neighbor libraries (FAISS, Annoy) on large datasets.

## Further reading

- [k-d tree (Wikipedia)](https://en.wikipedia.org/wiki/K-d_tree)
- [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [MNIST database](https://en.wikipedia.org/wiki/MNIST_database)
