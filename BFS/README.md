# Breadth-First Search & the "Six Degrees of Kevin Bacon"

> How few handshakes separate any two actors in Hollywood? This project answers that with a graph search algorithm written from scratch, then scales it to a real IMDb network.

| | |
|---|---|
| **Concepts** | Graphs, adjacency lists, queues, breadth-first search, shortest paths, bipartite graphs |
| **Stack** | Python, NetworkX, Matplotlib |
| **Code** | [`breadth_first_search.py`](breadth_first_search.py) |

---

## Why it's cool

Breadth-first search (BFS) is one of the simplest algorithms in computer science, yet it guarantees something powerful. In an unweighted network it always finds the **shortest** route between two points. It does this by exploring outward in rings: first every neighbor, then every neighbor's neighbor, and so on, the way a ripple spreads across a pond.

Point that at a database of movies and you get the famous **Bacon number**: the smallest number of shared films linking any actor to Kevin Bacon.

## Where it's used in the real world

- **Social networks:** "2nd- and 3rd-degree connections" on LinkedIn, and friend-of-friend suggestions.
- **Navigation and games:** shortest paths on grids and maps, and pathfinding for game AI on uniform-cost terrain.
- **Web crawlers:** visiting pages level by level outward from a seed URL.
- **Networking:** broadcast routing, and finding the fewest-hop route between routers.
- **Puzzles:** solving a Rubik's cube or a sliding-tile puzzle in the fewest moves.
- **Garbage collection and compilers:** finding every object or block reachable from a starting point.

## The concept in 60 seconds

A **graph** is a set of *nodes* (people, pages, cities) connected by *edges* (friendships, links, roads). This project stores a graph as an **adjacency dictionary**, which maps each node to the set of its neighbors:

```python
{'A': {'B', 'D'}, 'B': {'A', 'D'}, 'C': {'D'}, 'D': {'A', 'B', 'C'}}
```

BFS uses a **queue** (first in, first out):

1. Put the start node in the queue.
2. Take the front node, mark it visited, and add its unvisited neighbors to the back of the queue.
3. Repeat until the queue is empty or you reach the target.

Because nodes come off the queue in order of distance, the first time BFS reaches the target it has used the fewest possible edges. Recording each node's **parent** (the node it was discovered from) lets you walk backward from the target and rebuild the path.

**Cost:** each node and edge is handled about once, so BFS runs in $O(V + E)$ time with an efficient visited set.

## What's implemented

| Component | What it does |
|---|---|
| `Graph` | A from-scratch graph built on an adjacency dictionary: `add_node`, `add_edge`, `remove_node`, `remove_edge`, with `KeyError` handling for nodes that don't exist |
| `Graph.traverse(source)` | Breadth-first traversal that returns nodes in the order they were visited |
| `Graph.shortest_path(source, target)` | BFS with parent tracking, then path reconstruction from the target back to the source |
| `MovieGraph` | Loads IMDb data into a NetworkX **bipartite graph**, where movies and actors are both nodes and each edge means "acted in" |
| `MovieGraph.path_to_actor(a, b)` | The shortest actor-to-actor path and the degrees of separation. Movies are skipped when counting, which is why the path length is halved. |
| `MovieGraph.average_number(target)` | Every actor's distance to a target actor, plotted as a histogram, with the average returned |

**Design note:** in the bipartite graph, actors never connect to each other directly; they connect only through movies. A path Actor → Movie → Actor counts as **one** degree of separation, so the code divides the path length by 2.

## Results

Running the from-scratch `Graph` on a small example:

```python
>>> g = Graph({'A': {'B','D'}, 'B': {'A','D'}, 'C': {'D'}, 'D': {'A','B','C'}})
>>> g.traverse('A')
['A', 'D', 'B', 'C']
>>> g.shortest_path('A', 'C')
['A', 'D', 'C']
```

## Run it

```bash
cd BFS
python -i breadth_first_search.py
```

```python
>>> g = Graph({'A': {'B'}, 'B': {'A', 'C'}, 'C': {'B'}})
>>> g.shortest_path('A', 'C')
```

The `MovieGraph` part needs `movie_data.txt` in this folder. That IMDb extract isn't included in the repository. Each line lists a title followed by its cast, separated by `/`.

## Skills demonstrated

- Implementing a core algorithm from scratch, then checking it against an industry library (NetworkX).
- Data modeling: choosing a bipartite representation so a messy real-world dataset becomes a clean graph problem.
- Parsing and ingesting a large text dataset, and turning results into a histogram.

## Limitations & ideas to extend it

- `traverse` and `shortest_path` assume the graph is **connected**. If a node can't be reached they raise `IndexError`, because the queue runs out. A good exercise is to return an empty path, or raise a clear error, instead.
- The visited nodes are tracked in a list, so each "have I seen this?" check is $O(V)$. Switching to a `set` and `collections.deque` gives the true $O(V + E)$ running time.
- **Try it:** build bidirectional BFS, which searches from both ends at once, and measure the speed-up on the movie graph.

## Further reading

- [Breadth-first search (Wikipedia)](https://en.wikipedia.org/wiki/Breadth-first_search)
- [Six Degrees of Kevin Bacon](https://en.wikipedia.org/wiki/Six_Degrees_of_Kevin_Bacon)
- [Bipartite graphs](https://en.wikipedia.org/wiki/Bipartite_graph)
