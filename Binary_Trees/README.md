# Linked Lists, Binary Search Trees & AVL Trees

> Why do databases use trees? This project builds three data structures from scratch and benchmarks them, showing the gap between $O(n)$ and $O(\log n)$ in practice.

| | |
|---|---|
| **Concepts** | Linked lists, recursion, binary search trees, self-balancing trees, tree rotations, asymptotic complexity |
| **Stack** | Python, NumPy, Matplotlib, NetworkX (tree drawing) |
| **Code** | [`binary_trees.py`](binary_trees.py) |

---

## Why it's cool

A binary search tree (BST) can find any item among a million in about 20 steps, but only if it stays balanced. Insert data that's already sorted and the "tree" collapses into a straight line, no faster than a list.

The **AVL tree** fixes this. After each insert it checks its own shape and **rotates** nodes to stay balanced. This project demonstrates that failure and the fix.

## Where it's used in the real world

- **Databases and file systems:** B-trees and B+-trees, close relatives of BSTs, index nearly every relational database (PostgreSQL, MySQL) and file system (NTFS, ext4).
- **Language standard libraries:** C++ `std::map` and `std::set` and Java's `TreeMap` use balanced trees (red-black trees).
- **Operating systems:** Linux's process scheduler uses a red-black tree.
- **In-memory indexes:** ordered key-value stores, range queries, and autocomplete.
- **Interviews:** BSTs, rotations, and recursion are staples of technical interviews.

## The concept in 60 seconds

- **Singly linked list:** nodes in a chain, each pointing to the next. Searching means walking the chain: $O(n)$.
- **Binary search tree:** each node has up to two children. Everything in the left subtree is smaller, and everything in the right subtree is larger. A search discards half the remaining tree at each step, so it takes $O(\text{height})$.
- **The catch:** height depends on insertion order. Random data gives a height of about $\log_2 n$. Sorted data gives a height of $n - 1$.
- **AVL tree:** every node keeps a **balance factor** (right subtree height minus left subtree height) of -1, 0, or +1. When an insert breaks this rule, one of four rotations restores it: left-left, right-right, left-right, or right-left. Height is guaranteed to be $O(\log n)$.

## What's implemented

| Component | What it does |
|---|---|
| `SinglyLinkedList` | `append`, `iterative_find`, and `recursive_find` (the same search written two ways) |
| `BST.find` / `BST.insert` | Recursive search and insertion that keep track of each node's parent |
| `BST.remove` | Deletion handling **every case**: a leaf, a node with one child, a node with two children (swap with the in-order predecessor), and removing the root |
| `BST.__str__` / `BST.draw` | Prints the tree level by level, and draws it with NetworkX and Graphviz |
| `AVL(BST)` | Inherits from `BST` and rebalances from the inserted node up to the root |
| `AVL._rotate_*` | All four rotation cases, keeping parent pointers and the root reference correct |
| `prob4()` | Benchmarks **build** and **search** times for all three structures on $2^3$ to $2^{10}$ items, plotted on log-log axes |

## Results

Inserting 512 already-sorted keys, the worst case for a plain BST:

| Structure | Tree height | Steps to find the deepest item |
|---|---|---|
| `BST` | **511** (a straight line) | 512 |
| `AVL` | **9** (the minimum is $\lceil\log_2 513\rceil - 1 = 9$) | 10 |

The AVL tree is as short as a binary tree holding 512 items can possibly be.

## Run it

```bash
cd Binary_Trees
python -i binary_trees.py
```

```python
>>> bst, avl = BST(), AVL()
>>> for k in range(1, 16):
...     bst.insert(k); avl.insert(k)
>>> print(bst)   # one long chain
>>> print(avl)   # balanced, 4 levels
```

### Requirements and data
- `BST.draw()` needs Graphviz and `pygraphviz`.
- The benchmark `prob4()` reads `english.txt`, a word list that isn't included in the repository.

## Skills demonstrated

- Pointer-based data structures, with careful handling of edge cases (the root, one child, two children).
- Recursion, and inheritance (`AVL` extends `BST`).
- Empirical performance analysis: designing a benchmark and reading log-log plots.

## Ideas to extend it

- Add AVL deletion with rebalancing. It's disabled here; `remove()` raises `NotImplementedError`.
- Implement a red-black tree and compare how often each tree rotates.
- Add in-order, pre-order, and post-order traversal generators.

## Further reading

- [Binary search tree (Wikipedia)](https://en.wikipedia.org/wiki/Binary_search_tree)
- [AVL tree](https://en.wikipedia.org/wiki/AVL_tree), the first self-balancing BST (Adelson-Velsky & Landis, 1962)
- [B-tree](https://en.wikipedia.org/wiki/B-tree), the database cousin
