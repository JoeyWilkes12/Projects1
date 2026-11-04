"""Benchmark the solver and rank starting words.

    python benchmark.py            # about 10 minutes

Plays every possible answer (the 3,624-word madalynrose list) with each
strategy, then ranks opening words, and writes the results to ``results/``:

* ``strategies.csv``: win rate, average and worst guesses, and the
  distribution of guesses for each strategy.
* ``openers.csv``: every allowed guess scored as a first word (entropy,
  expected candidates left, heat-map score), plus full games for a
  shortlist of openers.

``Wordle_benchmark.ipynb`` reads these files and plots them.
"""
import os
import time

import numpy as np
import pandas as pd

from wordle_engine import best_guess, random_candidate, simulate, summarize
from wordle_solver import HERE, WordleSolver
from wordle_strategy import entropy, expected_remaining, heatmap_scores

RESULTS = os.path.join(HERE, "results")

STRATEGIES = {
    "entropy": dict(method="entropy"),
    "entropy (hard mode)": dict(method="entropy", hard_mode=True),
    "heatmap": dict(method="heatmap"),
    "frequency": dict(method="frequency"),
}

# Openers to play in full, besides the top entropy openers: popular human
# picks, and the vowel-heavy words hand-checked in the NLTK CSV's
# "In Word List" column.
POPULAR = ["crane", "slate", "raise", "stare", "arose", "trace"]
VOWEL_HEAVY = ["adieu", "audio", "ourie", "auloi", "yowie", "youse", "noyau"]
TOP_N = 10


def play(solver, strategy, first_guess=None):
    results = simulate(solver.words.tolist(), strategy, solver, first_guess=first_guess)
    stats = summarize(results)
    return {
        "win_rate": stats["win_rate"],
        "mean_guesses": stats["mean_guesses"],
        "worst": stats["max_guesses"],
        "distribution": " ".join(f"{k}:{v}" for k, v in stats["distribution"].items()),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    solver = WordleSolver()
    print(f"{len(solver.words)} answers, {len(solver.guess_words)} allowed guesses")

    # --- strategies -----------------------------------------------------------
    rows = []
    for name, kwargs in STRATEGIES.items():
        t = time.time()
        row = {"strategy": name, **play(solver, best_guess(**kwargs))}
        rows.append(row)
        print(f"{name}: {row['win_rate']:.2%} won, {row['mean_guesses']:.3f} guesses ({time.time() - t:.0f}s)")
    row = {"strategy": "random candidate", **play(solver, random_candidate(seed=12345))}
    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "strategies.csv"), index=False)

    # --- openers: one-step scores for every allowed guess -----------------------
    patterns = solver.patterns
    openers = pd.DataFrame(
        {
            "word": solver.guess_words,
            "entropy": entropy(patterns),
            "expected_left": expected_remaining(patterns),
            "heatmap": heatmap_scores(solver._guess_codes, solver._codes, solver._counts),
            "candidate": np.isin(solver.guess_words, solver.words),
            "frequency": solver.frequency,
        }
    ).sort_values("entropy", ascending=False, ignore_index=True)
    openers["entropy_rank"] = np.arange(1, len(openers) + 1)

    # --- openers: full games for a shortlist ------------------------------------
    shortlist = list(openers["word"].head(TOP_N)) + POPULAR + VOWEL_HEAVY
    shortlist = [w for w in dict.fromkeys(shortlist) if w in set(solver.guess_words)]
    played = []
    for word in shortlist:
        t = time.time()
        row = {"word": word, **play(solver, best_guess(), first_guess=word)}
        played.append(row)
        print(f"opener {word}: {row['mean_guesses']:.3f} guesses, {row['win_rate']:.2%} won ({time.time() - t:.0f}s)")
    openers = openers.merge(pd.DataFrame(played), on="word", how="left")
    openers.to_csv(os.path.join(RESULTS, "openers.csv"), index=False, float_format="%.4f")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
