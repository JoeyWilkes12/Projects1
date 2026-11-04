"""Wordle game engine and self-play.

The ``Wordle`` class is the module version of the class in
``wordle_engine.ipynb``. It holds a secret word, scores guesses with the
same ``score`` function the solver uses, and reports each result in the
solver's input notation (``sau?c!e``), so its output can be passed
straight to ``WordleSolver.add_guess``.

``self_play`` connects the two: a strategy picks each guess from the
solver's state, the engine scores it, and the solver narrows the
candidates, until the word is found or the guesses run out. ``simulate``
plays many games and summarizes them.

    >>> game = Wordle("butch")
    >>> game.guess("sauce")
    'sau?c!e'
    >>> self_play("shard", random_candidate(seed=0)).solved
    True
"""
import random
from dataclasses import dataclass, field

from wordle_solver import ALPHABET, WordleSolver, load_words, score, to_notation


class Wordle:
    """One game of Wordle.

    ``wordle``: the secret word. If it's ``None``, a random word is chosen
    from ``words`` (the NLTK list by default), using ``seed`` if given.

    ``allowed``: if given, guesses must be in this collection of words, as in
    the real game. By default any string of the right length is accepted.
    """

    def __init__(self, wordle=None, words=None, max_guesses=6, allowed=None, seed=None):
        if wordle is None:
            wordle = random.Random(seed).choice(load_words(words))
        wordle = wordle.strip().lower()
        if not wordle or any(c not in ALPHABET for c in wordle):
            raise ValueError(f"the wordle must be letters a-z, got {wordle!r}")
        self.wordle = wordle
        self.max_guesses = max_guesses
        self.allowed = set(allowed) if allowed is not None else None
        self.history = []  # list of (guess, pattern)

    @classmethod
    def from_hidden_input(cls, **kwargs):
        """Ask for the wordle without echoing it, so one person can set a
        puzzle for another (as in the notebook)."""
        from getpass import getpass

        return cls(getpass("Please enter in the wordle\n"), **kwargs)

    # ----- state -----------------------------------------------------------

    @property
    def solved(self):
        return bool(self.history) and self.history[-1][0] == self.wordle

    @property
    def guesses_left(self):
        return self.max_guesses - len(self.history)

    @property
    def over(self):
        return self.solved or self.guesses_left <= 0

    # ----- playing ---------------------------------------------------------

    def check(self, guess):
        """Raise ``ValueError`` if ``guess`` isn't a valid guess; otherwise
        return it lowercased. An invalid guess doesn't use up a turn."""
        guess = guess.strip().lower()
        if len(guess) != len(self.wordle):
            raise ValueError(f"guess needs to have {len(self.wordle)} letters")
        bad = set(guess) - set(ALPHABET)
        if bad:
            raise ValueError(f"the following characters are not allowed: {sorted(bad)}")
        if self.allowed is not None and guess not in self.allowed:
            raise ValueError(f"{guess!r} is not in the word list")
        return guess

    def guess(self, guess):
        """Score one guess and return it in the solver's notation."""
        if self.over:
            raise ValueError("the game is over")
        guess = self.check(guess)
        pattern = score(guess, self.wordle)
        self.history.append((guess, pattern))
        return to_notation(guess, pattern)

    def play(self, guesses=None, input_fn=input, output_fn=print):
        """Play until the word is found or the guesses run out, and return
        ``True`` if it was found. Guesses are typed at a prompt, or taken
        from the ``guesses`` list to replay a game."""
        guesses = iter(guesses) if guesses is not None else None
        while not self.over:
            if guesses is None:
                attempt = input_fn("Submit a guess - ")
            else:
                attempt = next(guesses, None)
                if attempt is None:
                    return False  # ran out of scripted guesses
                output_fn(f"Submit a guess - {attempt}")
            try:
                output_fn(self.guess(attempt))
            except ValueError as e:
                output_fn(e)  # a typo doesn't end the game or use up a guess
        if self.solved:
            output_fn(f"you did it in {len(self.history)}")
        else:
            output_fn(f"out of guesses - the wordle was {self.wordle}")
        return self.solved


# ----- self-play ----------------------------------------------------------------


@dataclass
class GameResult:
    answer: str
    guesses: list = field(default_factory=list)  # feedback notation, one per guess
    candidates: list = field(default_factory=list)  # candidates left after each guess
    solved: bool = False

    @property
    def n_guesses(self):
        return len(self.guesses)


def random_candidate(seed=None):
    """Strategy: guess a random word that could still be the answer.

    This is v0's planned ``random_solver``: no lookahead, just "pick
    anything that fits", which makes it a useful baseline.
    """
    rng = random.Random(seed)

    def choose(solver):
        return rng.choice(solver.remaining_words)

    return choose


def self_play(answer, strategy, solver=None, max_guesses=6, first_guess=None):
    """Play one game: ``strategy(solver)`` picks each guess, the engine
    scores it, and the solver narrows the candidates.

    ``first_guess`` fixes the opening word. ``solver`` can be passed in to
    reuse its loaded word list; it is reset first.
    """
    solver = solver if solver is not None else WordleSolver()
    solver.reset()
    game = Wordle(answer, max_guesses=max_guesses)
    result = GameResult(answer)
    while not game.over:
        guess = first_guess if (first_guess and not game.history) else strategy(solver)
        notation = game.guess(guess)
        remaining = solver.add_guess(notation)
        result.guesses.append(notation)
        result.candidates.append(len(remaining))
        if answer not in remaining:  # can only happen if the solver is wrong
            raise AssertionError(f"solver dropped the answer {answer!r} after {result.guesses}")
    result.solved = game.solved
    return result


def simulate(answers, strategy, solver=None, max_guesses=6, first_guess=None):
    """Play one game per answer and return a pandas DataFrame with a row per game.

    Games aren't cut off at ``max_guesses``: the solver keeps going until it
    finds the word, so the table also shows how far over the limit a lost
    game would have run. ``solved`` is ``True`` for games won within the limit.
    """
    import pandas as pd

    solver = solver if solver is not None else WordleSolver()
    rows = []
    for answer in answers:
        r = self_play(answer, strategy, solver, max_guesses=len(solver.words), first_guess=first_guess)
        rows.append(
            {
                "answer": answer,
                "guesses": r.n_guesses,
                "solved": r.n_guesses <= max_guesses,
                "first_guess": r.guesses[0],
                "history": " ".join(r.guesses),
            }
        )
    return pd.DataFrame(rows)


def summarize(results, max_guesses=6):
    """Win rate, average guesses, and the distribution of guesses per game."""
    dist = results["guesses"].value_counts().sort_index()
    return {
        "games": len(results),
        "win_rate": float(results["solved"].mean()),
        "mean_guesses": float(results["guesses"].mean()),
        "max_guesses": int(results["guesses"].max()),
        "distribution": {int(k): int(v) for k, v in dist.items()},
    }
