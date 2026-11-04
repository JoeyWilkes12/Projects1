"""Tests for wordle_engine.py.  Run from the repo root or this folder:

    python -m pytest Wordle
"""
import os

import pytest

from wordle_engine import Wordle, random_candidate, self_play, simulate, summarize
from wordle_solver import WordleSolver, load_words, parse_feedback

HERE = os.path.dirname(os.path.abspath(__file__))
MADALYNROSE = load_words(os.path.join(HERE, "madalynrose list of 5 letter words parsed v0.csv"))

# Real games: the answer and the feedback Wordle showed, as typed.
# Add more from your screenshots here; each one is checked against the engine.
REAL_GAMES = [
    ("butch", ["sau?c!e", "hu!nc!h!", "mu!lc!h!", "du!t!c!h!", "b!u!t!c!h!"]),
    ("shard", ["wea!r!y", "apa!r!t", "boa!r!d!", "gua!r!d!", "s!h!a!r!d!"]),
    ("mount", ["learn?", "so!u!n!d", "t?icks", "m!o!u!n!t!"]),
    ("sugar", ["ca?ndy", "a?fter!", "s!ola!r!", "s!u!g!a!r!"]),
    ("inert", ["yar?n?s", "hon?or?", "n?e?r?ve", "i!n!ur!e?", "i!n!e!r!t!"]),
]


# ----- the engine ----------------------------------------------------------------

@pytest.mark.parametrize("answer, notations", REAL_GAMES)
def test_engine_reproduces_real_games(answer, notations):
    game = Wordle(answer)
    for notation in notations:
        guess, _ = parse_feedback(notation)
        assert game.guess(guess) == notation
    assert game.solved


@pytest.mark.parametrize(
    "answer, guess, expected",
    [
        ("vivid", "vviiv", "v!v?i?i!v"),
        ("vivid", "ivoiv", "i?v?oi!v?"),
        ("eerie", "leaee", "le!ae?e!"),
        ("eerie", "lease", "le!ase!"),
    ],
)
def test_engine_repeated_letters(answer, guess, expected):
    assert Wordle(answer).guess(guess) == expected


@pytest.mark.parametrize("bad", ["ba", "3acba", "cactus", "ab de"])
def test_invalid_guess_does_not_count(bad):
    game = Wordle("vivid")
    with pytest.raises(ValueError):
        game.guess(bad)
    assert game.guesses_left == 6


def test_allowed_list():
    game = Wordle("shard", allowed={"shard", "weary"})
    with pytest.raises(ValueError, match="not in the word list"):
        game.guess("xxxxx")
    assert game.guess("WEARY") == "wea!r!y"  # case-insensitive


def test_six_misses_lose():
    game = Wordle("shard")
    for _ in range(6):
        game.guess("weary")
    assert game.over and not game.solved
    with pytest.raises(ValueError, match="over"):
        game.guess("shard")


def test_random_wordle_is_reproducible():
    assert Wordle(seed=7).wordle == Wordle(seed=7).wordle
    assert Wordle(words=["shard"]).wordle == "shard"


def test_play_replays_a_game():
    out = []
    won = Wordle("butch").play(["sauce", "oops", "hunch", "mulch", "dutch", "butch"], output_fn=out.append)
    assert won
    assert out[-1] == "you did it in 5"  # the typo "oops" didn't count
    lost = Wordle("butch").play(["sauce"] * 6, output_fn=out.append)
    assert not lost and out[-1] == "out of guesses - the wordle was butch"


# ----- self-play ----------------------------------------------------------------

@pytest.fixture(scope="module")
def mr_solver():
    return WordleSolver(MADALYNROSE)


def test_self_play_one_game(mr_solver):
    r = self_play("shard", random_candidate(seed=1), mr_solver, first_guess="weary")
    assert r.guesses[0] == "wea!r!y"
    assert r.candidates == sorted(r.candidates, reverse=True)  # never grows
    if r.solved:
        assert r.guesses[-1] == "s!h!a!r!d!"


def test_self_play_every_answer(mr_solver):
    """Plays all 3,624 words in the madalynrose list. self_play raises if the
    solver ever drops the answer, so this also checks the solver at scale."""
    results = simulate(MADALYNROSE, random_candidate(seed=0), mr_solver)
    stats = summarize(results)
    assert stats["games"] == len(MADALYNROSE)
    assert (results["history"].str.split().str[-1] == results["answer"].map(lambda a: "!".join(a) + "!")).all()
    assert stats["win_rate"] > 0.9  # random guessing among candidates: about 94%
