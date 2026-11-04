"""Tests for wordle_strategy.py and WordleSolver.suggest.

    python -m pytest Wordle
"""
import random

import numpy as np
import pytest

from wordle_engine import best_guess, simulate, summarize
from wordle_solver import WordleSolver, load_words, score
from wordle_strategy import encode, entropy, heatmap_scores, letter_counts, pattern_matrix

NLTK = load_words()


def code(pattern):
    return sum({"b": 0, "y": 1, "g": 2}[c] * 3**i for i, c in enumerate(pattern))


@pytest.fixture(scope="module")
def solver():
    """Default lists: 3,624 possible answers, 9,795 allowed guesses."""
    s = WordleSolver()
    s.suggest(1)  # builds the pattern table once (about 7 s)
    return s


# ----- the pattern table -------------------------------------------------------------

def test_pattern_matrix_matches_score():
    rng = random.Random(0)
    guesses, answers = rng.sample(NLTK, 300), rng.sample(NLTK, 300)
    p = pattern_matrix(encode(guesses), encode(answers))
    for _ in range(5000):
        i, j = rng.randrange(300), rng.randrange(300)
        assert p[i, j] == code(score(guesses[i], answers[j]))


@pytest.mark.parametrize("guess, answer", [("vviiv", "vivid"), ("speed", "abide"), ("geese", "those"), ("eerie", "eerie"), ("llama", "hello")])
def test_pattern_matrix_repeated_letters(guess, answer):
    assert pattern_matrix(encode([guess]), encode([answer]))[0, 0] == code(score(guess, answer))


# ----- scoring methods ------------------------------------------------------------------

def test_entropy_simple_cases():
    patterns = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 2, 3]], dtype=np.uint8)
    assert np.allclose(entropy(patterns), [0, 1, 2])  # no split, halves, quarters
    assert np.allclose(entropy(patterns[:, :1]), 0)  # one candidate: nothing to learn


def test_heatmap_scores():
    candidates = ["shard", "shark"]
    codes = encode(candidates)
    scores = heatmap_scores(encode(["shard", "zzzzz"]), codes, letter_counts(codes))
    # shard: s, h, a, r all certain (2 each); d half (0.5 + 0.5)
    assert scores[0] == pytest.approx(4 * 2 + 1.0)
    assert scores[1] == 0


# ----- suggest ------------------------------------------------------------------------

def test_opening_suggestion(solver):
    solver.reset()
    top = solver.suggest(3)
    assert list(top.columns) == ["word", "score", "candidate", "frequency"]
    assert top["score"].is_monotonic_decreasing
    assert top["word"].iloc[0] == "tares"  # best opener for these lists


def test_last_candidate_is_suggested(solver):
    solver.solve_from_history(["wea!r!y", "apa!r!t", "boa!r!d!", "gua!r!d!"])
    assert solver.remaining_words == ["shard"]
    assert solver.suggest(1)["word"].iloc[0] == "shard"


def test_two_candidates_prefers_a_candidate(solver):
    solver.solve_from_history(["wea!r!y", "apa!r!t", "boa!r!d!"])
    remaining = solver.remaining_words
    assert remaining == ["guard", "shard"]
    assert solver.suggest(1)["word"].iloc[0] in remaining  # a 50% chance to win now


def test_hard_mode_only_suggests_consistent_words(solver):
    solver.solve_from_history(["wea!r!y"])
    check = WordleSolver(solver.guess_words.tolist())
    check.solve_from_history(["wea!r!y"])
    consistent = set(check.remaining_words)
    for method in ("entropy", "heatmap"):
        assert set(solver.suggest(20, method=method, hard_mode=True)["word"]) <= consistent


def test_frequency_breaks_ties():
    s = WordleSolver(["sharp", "shard", "shark"])
    s.add_guess("s!h!a!r!x")  # all three remain
    top = s.suggest(3, method="frequency")
    assert top["frequency"].is_monotonic_decreasing
    assert set(top["word"]) == {"sharp", "shard", "shark"}


def test_bad_method(solver):
    solver.reset()
    with pytest.raises(ValueError, match="unknown method"):
        solver.suggest(method="vibes")


def test_no_candidates_left():
    s = WordleSolver(["shard"])
    s.add_guess("shark")  # typed as all gray, which rules out shard
    with pytest.raises(ValueError, match="no possible answers"):
        s.suggest()


# ----- self-play with suggestions ------------------------------------------------------

def test_entropy_strategy_wins(solver):
    answers = random.Random(42).sample(solver.words.tolist(), 60)
    stats = summarize(simulate(answers, best_guess(), solver))
    assert stats["win_rate"] == 1.0
    assert stats["mean_guesses"] < 4


def test_first_guess_parameter(solver):
    results = simulate(["shard", "mount"], best_guess(), solver, first_guess="crane")
    assert (results["first_guess"].str.replace("!", "").str.replace("?", "") == "crane").all()
