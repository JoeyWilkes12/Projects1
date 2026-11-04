# Markov Chains: Forecasting Weather & Babbling Like Yoda

> Predict tomorrow from today, find where a random process settles in the long run, and generate surprisingly convincing (and funny) sentences, all from one idea: **the future depends only on the present.**

| | |
|---|---|
| **Concepts** | Stochastic processes, transition matrices, steady-state distributions, simulation, text generation |
| **Stack** | NumPy, SciPy, pytest |
| **Code** | [`markov_chains.py`](markov_chains.py), with tests in [`test_markov.py`](test_markov.py) |

---

## Why it's cool

A Markov chain is a system that hops between states with fixed probabilities. Today is hot, so there's a 70% chance tomorrow is hot too. That's the whole model, yet it answers deep questions:
- What fraction of days will be hot in the long run?
- What does a typical sentence look like if each word only remembers the word before it?

Train the sentence generator on Yoda's lines and it produces new ones in his voice. A sample saved in the code:

> *"mourn them do not, miss them do not"*

## Where it's used in the real world

- **Language:** autocomplete and predictive text. N-gram language models were the forerunners of today's LLMs.
- **Web search:** PageRank is the steady state of a Markov chain (see [`../PageRank`](../PageRank)).
- **Finance:** credit-rating transitions, and regime-switching market models.
- **Biology:** DNA sequence analysis, and hidden Markov models for gene finding.
- **Operations:** queueing systems, inventory, and customer churn and lifetime value.
- **Statistics and AI:** MCMC sampling for Bayesian inference, and reinforcement learning (Markov decision processes).

## The concept in 60 seconds

**Transition matrix.** $A_{ij}$ is the probability of moving from state $j$ to state $i$. Every column sums to 1; this is called **column-stochastic**.

**Forecasting.** If $\mathbf{x}_t$ is today's probability distribution over states, then tomorrow's is $\mathbf{x}_{t+1} = A\mathbf{x}_t$.

**Steady state.** Keep multiplying and, for well-behaved chains, $\mathbf{x}_t$ converges to a vector $\mathbf{x}^*$ with $A\mathbf{x}^* = \mathbf{x}^*$. That is an eigenvector with eigenvalue 1, and it's the long-run share of time spent in each state, whatever the starting state.

**Text generation.** Every unique word is a state, plus special `$tart` and `$top` states. Count which word follows which in a corpus, normalize each column, then random-walk from `$tart` until you hit `$top`.

## What's implemented

| Function / class | What it does |
|---|---|
| `random_chain(n)` | Generates a random $n \times n$ column-stochastic matrix using NumPy broadcasting |
| `forecast(n)` | Simulates $n$ days of a 2-state (hot/cold) weather chain with binomial draws |
| `four_state_forecast(days)` | A 4-state chain (hot, mild, cold, freezing) simulated with multinomial draws |
| `steady_state(A, tol, N)` | Iterates $\mathbf{x} \leftarrow A\mathbf{x}$ from a random start; raises `ValueError` if it doesn't converge |
| `SentenceGenerator(filename)` | Builds a word-transition matrix from a text file (one sentence per line) |
| `SentenceGenerator.babble()` | Random-walks the chain to produce a new sentence |

## Results

- **2-state weather:** the steady state is $[2/3,\ 1/3]$. In the long run two out of three days are hot, which matches the analytic answer.
- **4-state weather:**

  | State | Hot | Mild | Cold | Freezing |
  |---|---|---|---|---|
  | Share of days | 24.7% | 30.0% | 33.3% | 12.1% |

- **Tests:** `pytest test_markov.py` runs **4 tests**, all passing:
  - every column of `random_chain` sums to 1;
  - $A\mathbf{x}^* = \mathbf{x}^*$ for random chains, and the known 2×2 steady state is recovered;
  - over 10,000 simulated days, the share of cold days is within 1 percentage point of the steady-state value of 1/3;
  - over 100,000 simulated days of the 4-state chain, the share of each weather type is within 0.5 percentage points of the steady state.

  These tests connect simulation to theory: the long-run frequencies from simulating the chain match the eigenvector.

## Run it

```bash
cd Markov_Chains
python -m pytest test_markov.py
```

```bash
python -i markov_chains.py
```

```python
>>> steady_state(random_chain(5))
>>> four_state_forecast(10)
>>> SentenceGenerator("your_corpus.txt").babble()   # any text, one sentence per line
```

Corpus files such as `yoda.txt` aren't included in the repository; any plain-text file works. [Project Gutenberg](https://www.gutenberg.org/) is a good source.

## Skills demonstrated

- Probabilistic modeling and Monte Carlo simulation.
- Iterative numerical methods with tolerance checks and failure modes (`ValueError` when the iteration doesn't converge).
- Writing unit tests with `pytest` fixtures for code that involves randomness.
- A first look at NLP: tokenizing, building a vocabulary, and generative language models.

## Ideas to extend it

- Use a **2nd-order** chain, where the state is the previous two words, for more coherent sentences.
- Store transitions in `scipy.sparse`. A real vocabulary makes the dense matrix huge and almost entirely zeros.
- Compute the steady state as an eigenvector with `np.linalg.eig` and compare it with iteration.

## Further reading

- [Markov chain (Wikipedia)](https://en.wikipedia.org/wiki/Markov_chain)
- [Stationary distribution](https://en.wikipedia.org/wiki/Discrete-time_Markov_chain#Stationary_distributions_relation_to_eigenvectors_and_simplices)
- Claude Shannon, *A Mathematical Theory of Communication* (1948), which generated English text with Markov chains
