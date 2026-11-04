"""Tests for wordle_solver.py.  Run from the repo root or this folder:

    python -m pytest Wordle
"""
import random

import pytest

from wordle_solver import (
    Constraints,
    WordleSolver,
    letter_count_table,
    load_words,
    parse_feedback,
    score,
    to_notation,
)

WORDS = load_words()  # the NLTK list, 8,497 words


@pytest.fixture(scope="module")
def solver():
    return WordleSolver(WORDS)


def brute_force(history, words=WORDS):
    """Words that would have produced exactly these colors for every guess."""
    return sorted(w for w in words if all(score(g, w) == p for g, p in history))


# ----- scoring -------------------------------------------------------------

@pytest.mark.parametrize(
    "guess, answer, expected",
    [
        ("sauce", "butch", "bbygb"),
        ("butch", "butch", "ggggg"),
        ("vviiv", "vivid", "gyygb"),  # third v is gray: vivid only has two
        ("ivoiv", "vivid", "yybgy"),
        ("leaee", "eerie", "bgbyg"),
        ("lease", "eerie", "bgbbg"),
        ("speed", "abide", "bbyby"),  # first e yellow, second e gray
        ("geese", "those", "bbbgg"),  # only one e left after the green
    ],
)
def test_score(guess, answer, expected):
    assert score(guess, answer) == expected


# ----- notation ------------------------------------------------------------

def test_parse_feedback_round_trip():
    for notation in ["wea!r!y", "learn?", "s!h!a!r!d!", "t?icks"]:
        assert to_notation(*parse_feedback(notation)) == notation
    assert parse_feedback("WEA!R!Y") == ("weary", "bbggb")


@pytest.mark.parametrize("bad", ["", "!abcd", "a!!bcd", "a?!bcd", "ab3de", "ab de"])
def test_parse_feedback_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_feedback(bad)


def test_wrong_length_rejected(solver):
    solver.reset()
    with pytest.raises(ValueError, match="expected 5 letters"):
        solver.add_guess("cat?s!")  # four letters


# ----- contradictions --------------------------------------------------------

def test_contradictory_greens():
    c = Constraints()
    c.add("shard", "ggggg")
    with pytest.raises(ValueError, match="slot 1"):
        c.add("chard", "ggggg")


def test_contradictory_counts():
    c = Constraints()
    c.add("eerie", "bbbbb")  # no e at all
    with pytest.raises(ValueError, match="at least 1"):
        c.add("lease", "bybbb")  # ...but now an e is yellow


# ----- exactness: the solver keeps exactly the consistent words ----------------

def test_matches_brute_force_on_random_games(solver):
    rng = random.Random(0)
    for _ in range(150):
        answer = rng.choice(WORDS)
        history = []
        solver.reset()
        for _ in range(rng.randint(1, 3)):
            guess = rng.choice(WORDS)
            history.append((guess, score(guess, answer)))
            remaining = solver.add_guess(to_notation(*history[-1]))
        assert answer in remaining
        assert remaining == brute_force(history)


def test_matches_brute_force_with_repeated_letters(solver):
    """Guesses and answers chosen to have repeated letters, where v0 kept extra words."""
    rng = random.Random(1)
    doubles = [w for w in WORDS if len(set(w)) < 5]
    for _ in range(100):
        answer = rng.choice(WORDS)
        history = [(g, score(g, answer)) for g in rng.sample(doubles, 2)]
        remaining = solver.solve_from_history([to_notation(g, p) for g, p in history])
        assert answer in remaining
        assert remaining == brute_force(history)


def test_one_at_a_time_equals_whole_history(solver):
    notations = ["wea!r!y", "apa!r!t", "boa!r!d!"]
    solver.reset()
    for n in notations:
        step_by_step = solver.add_guess(n)
    assert step_by_step == solver.solve_from_history(notations)


# ----- real games (from test_class() in Wordle_Solver_v0.py) -------------------

@pytest.mark.parametrize(
    "answer, notations",
    [
        ("shard", ["wea!r!y", "apa!r!t", "boa!r!d!", "gua!r!d!", "s!h!a!r!d!"]),
        ("mount", ["learn?", "so!u!n!d", "t?icks", "m!o!u!n!t!"]),
        ("sugar", ["ca?ndy", "a?fter!", "s!ola!r!", "s!u!g!a!r!"]),
        ("inert", ["yar?n?s", "hon?or?", "n?e?r?ve", "i!n!ur!e?", "i!n!e!r!t!"]),
    ],
)
def test_real_games(solver, answer, notations):
    history = []
    for n in notations:
        history.append(parse_feedback(n))
        # the typed feedback matches what Wordle would have shown
        assert score(history[-1][0], answer) == history[-1][1]
    remaining = solver.solve_from_history(notations)
    assert remaining == [answer]
    assert solver.is_solved


def test_v0_keeps_words_v1_removes():
    """A case v0 got wrong: 'sheep' against 'skate' leaves exactly one e."""
    history = [("sheep", score("sheep", "skate"))]  # 'gbbyb': second e gray
    remaining = WordleSolver(WORDS).solve_from_history([to_notation(*history[0])])
    assert "skate" in remaining
    assert all(w.count("e") == 1 for w in remaining)
    assert remaining == brute_force(history)


# ----- data loading ------------------------------------------------------------

def test_load_words_repairs_false():
    assert "false" in WORDS
    assert all(w.islower() and len(w) == 5 for w in WORDS)


def test_letter_count_table():
    table = letter_count_table(["eerie", "shard"])
    assert table.loc["eerie", "e"] == 3
    assert table.loc["shard", "a"] == 1
    assert table.loc["shard", "e"] == 0
