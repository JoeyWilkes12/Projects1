"""Wordle solver, v1: exact constraint filtering over a word list.

Feedback is typed in the same notation as v0 (``Wordle_Solver_v0.py``):
each letter of the guess, followed by ``!`` if it came back green or ``?``
if it came back yellow. Gray letters are left bare::

    >>> solver = WordleSolver(["shard", "board", "guard", "weary"])
    >>> solver.add_guess("wea!r!y")
    ['board', 'guard', 'shard']

What changed from v0:

* **Letter counts, not just presence.** Each piece of feedback becomes a
  minimum and, where a copy came back gray, a maximum count for that
  letter, plus the slots the letter can't be in. The remaining words are
  then exactly the words that would have produced the same colors for
  every guess so far. ``test_wordle_solver.py`` checks this against brute
  force.
* **Whole games in one call.** ``solve_from_history`` takes a list of
  guesses and filters the word list once.
* **Vectorized filtering.** Words are stored as NumPy arrays of letter
  codes and letter counts, so each constraint is one array comparison.
* **Input checks.** Badly typed feedback, and feedback that contradicts an
  earlier guess, raise ``ValueError`` with a message saying what's wrong.
"""
import os
import string
from collections import Counter

import numpy as np

GREEN, YELLOW, GRAY = "g", "y", "b"
ALPHABET = string.ascii_lowercase
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORDS_CSV = os.path.join(HERE, "5 letter words parsed - NLTK Words Corpora - v0.csv")


def score(guess, answer):
    """Return Wordle's colors for ``guess`` against ``answer``.

    The result is a string of ``g`` (green), ``y`` (yellow) and ``b`` (gray),
    one per letter. Repeated letters follow Wordle's rules: greens are marked
    first and use up their letter, then yellows are handed out from the
    letters that are left, from left to right.

    >>> score("sauce", "butch")
    'bbygb'
    >>> score("vviiv", "vivid")
    'gyygb'
    """
    if len(guess) != len(answer):
        raise ValueError(f"guess {guess!r} and answer {answer!r} have different lengths")
    result = [GRAY] * len(guess)
    unmatched = Counter()
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = GREEN
        else:
            unmatched[a] += 1
    for i, g in enumerate(guess):
        if result[i] == GRAY and unmatched[g] > 0:
            result[i] = YELLOW
            unmatched[g] -= 1
    return "".join(result)


def parse_feedback(notation):
    """Split feedback notation into ``(guess, pattern)``.

    >>> parse_feedback("wea!r!y")
    ('weary', 'bbggb')
    >>> parse_feedback("learn?")
    ('learn', 'bbbby')
    """
    notation = notation.strip().lower()
    guess, pattern = [], []
    for ch in notation:
        if ch in ALPHABET:
            guess.append(ch)
            pattern.append(GRAY)
        elif ch in "!?":
            if not guess or pattern[-1] != GRAY:
                raise ValueError(
                    f"{notation!r}: '{ch}' must follow a letter, and each letter takes at most one mark"
                )
            pattern[-1] = GREEN if ch == "!" else YELLOW
        else:
            raise ValueError(f"{notation!r}: unexpected character {ch!r}; use letters, '!' and '?'")
    if not guess:
        raise ValueError("feedback is empty")
    return "".join(guess), "".join(pattern)


def to_notation(guess, pattern):
    """Inverse of ``parse_feedback``: ``('weary', 'bbggb')`` -> ``'wea!r!y'``."""
    marks = {GREEN: "!", YELLOW: "?", GRAY: ""}
    return "".join(letter + marks[color] for letter, color in zip(guess, pattern))


class Constraints:
    """Everything known about the answer, built up one guess at a time.

    * ``greens[slot]``: the letter known to be in that slot.
    * ``banned[slot]``: letters known *not* to be in that slot.
    * ``min_count[letter]``: the answer has at least this many copies.
    * ``max_count[letter]``: the answer has at most this many copies.
      A gray copy of a letter caps its count at the number of green and
      yellow copies in the same guess (zero if there are none).
    """

    def __init__(self, length=5):
        self.length = length
        self.greens = {}
        self.banned = {slot: set() for slot in range(length)}
        self.min_count = {}
        self.max_count = {}

    def add(self, guess, pattern):
        """Add one guess and its colors; raise ``ValueError`` on a contradiction."""
        if len(guess) != self.length or len(pattern) != self.length:
            raise ValueError(f"{guess!r}: expected {self.length} letters, got {len(guess)}")
        found = Counter()  # green + yellow copies of each letter in this guess
        has_gray = set()
        for slot, (letter, color) in enumerate(zip(guess, pattern)):
            if color == GREEN:
                if self.greens.get(slot, letter) != letter:
                    raise ValueError(
                        f"slot {slot + 1} was green for {self.greens[slot]!r} earlier, "
                        f"but green for {letter!r} in {guess!r}"
                    )
                self.greens[slot] = letter
                found[letter] += 1
            else:
                # yellow or gray: this letter is not in this slot
                self.banned[slot].add(letter)
                if color == YELLOW:
                    found[letter] += 1
                else:
                    has_gray.add(letter)
        for letter, n in found.items():
            self.min_count[letter] = max(self.min_count.get(letter, 0), n)
        for letter in has_gray:
            self.max_count[letter] = min(self.max_count.get(letter, self.length), found[letter])
        self._check()

    def _check(self):
        for letter, low in self.min_count.items():
            high = self.max_count.get(letter, self.length)
            if low > high:
                raise ValueError(
                    f"contradictory feedback: {letter!r} needs at least {low} "
                    f"but at most {high} copies"
                )
        for slot, letter in self.greens.items():
            if letter in self.banned[slot]:
                raise ValueError(
                    f"contradictory feedback: {letter!r} was green in slot {slot + 1} "
                    f"but yellow or gray there in another guess"
                )
        if sum(self.min_count.values()) > self.length:
            raise ValueError("contradictory feedback: more confirmed letters than slots")

    def allows(self, word):
        """Pure-Python check of one word. Slower than ``WordleSolver``'s
        vectorized filter, but easy to read; the tests use it as a reference."""
        if len(word) != self.length:
            return False
        if any(word[slot] != letter for slot, letter in self.greens.items()):
            return False
        if any(word[slot] in letters for slot, letters in self.banned.items()):
            return False
        counts = Counter(word)
        if any(counts[letter] < n for letter, n in self.min_count.items()):
            return False
        return all(counts[letter] <= n for letter, n in self.max_count.items())


def load_words(source=None, length=5):
    """Load a word list as a sorted list of unique lowercase strings.

    ``source`` may be a path to one of the CSVs in this folder (any CSV with
    a ``word`` column), a pandas DataFrame with a ``word`` column, or any
    iterable of strings. The default is the NLTK CSV next to this file.
    Words that aren't ``length`` letters a-z are dropped.
    """
    if source is None:
        source = DEFAULT_WORDS_CSV
    if isinstance(source, str):
        import pandas as pd

        # keep_default_na=False stops pandas reading words like "null" as NaN
        source = pd.read_csv(source, usecols=["word"], dtype=str, keep_default_na=False)
    if hasattr(source, "columns"):
        source = source["word"]
    # .lower() also repairs "FALSE": a spreadsheet stored the word "false" as a boolean
    words = {str(w).strip().lower() for w in source}
    return sorted(w for w in words if len(w) == length and all(c in ALPHABET for c in w))


def letter_count_table(words):
    """A DataFrame with one row per word and one column per letter a-z,
    holding how many times that letter appears.

    This is the count-based version of the yes/no letter columns in the
    CSVs (the "number of times each letter appears" item in IDEAS.md).
    """
    import pandas as pd

    words = list(words)
    codes = np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int8)
    counts = np.zeros((len(words), 26), dtype=np.int8)
    for slot in range(codes.shape[1]):
        np.add.at(counts, (np.arange(len(words)), codes[:, slot]), 1)
    return pd.DataFrame(counts, index=pd.Index(words, name="word"), columns=list(ALPHABET))


class WordleSolver:
    """Narrows a word list down using Wordle feedback.

    >>> solver = WordleSolver()                    # NLTK word list
    >>> remaining = solver.add_guess("wea!r!y")    # one guess at a time...
    >>> remaining = solver.solve_from_history(["wea!r!y", "apa!r!t"])  # ...or a whole game
    """

    def __init__(self, words=None, wordle_length=5):
        self.wordle_length = wordle_length
        self.words = np.array(load_words(words, wordle_length))
        if len(self.words) == 0:
            raise ValueError(f"no {wordle_length}-letter words in the word list")
        # letter codes (0-25) for every slot, and letter counts, one row per word
        self._codes = (
            np.frombuffer("".join(self.words).encode("ascii"), dtype=np.uint8)
            .reshape(len(self.words), wordle_length)
            .astype(np.int8)
            - 97
        )
        self._counts = np.zeros((len(self.words), 26), dtype=np.int8)
        for slot in range(wordle_length):
            np.add.at(self._counts, (np.arange(len(self.words)), self._codes[:, slot]), 1)
        self.reset()

    def reset(self):
        """Forget all guesses."""
        self.constraints = Constraints(self.wordle_length)
        self.history = []  # list of (guess, pattern)
        self._remaining = np.arange(len(self.words))

    # ----- feedback --------------------------------------------------------

    def add_guess(self, guess_results):
        """Add one guess with its feedback (``"wea!r!y"``) and return the
        remaining words. Only the words still remaining are re-checked."""
        guess, pattern = parse_feedback(guess_results)
        self.constraints.add(guess, pattern)  # raises on bad length or contradiction
        self.history.append((guess, pattern))
        self._remaining = self._remaining[self._mask(self._remaining)]
        return self.remaining_words

    def solve_from_history(self, guesses_results):
        """Start over, apply a whole list of feedback strings, and filter the
        word list once at the end. Returns the remaining words."""
        self.reset()
        for gr in guesses_results:
            guess, pattern = parse_feedback(gr)
            self.constraints.add(guess, pattern)
            self.history.append((guess, pattern))
        self._remaining = self._remaining[self._mask(self._remaining)]
        return self.remaining_words

    def _mask(self, rows):
        """Boolean mask over ``rows`` of the words that satisfy every constraint."""
        c = self.constraints
        codes, counts = self._codes[rows], self._counts[rows]
        mask = np.ones(len(rows), dtype=bool)
        for slot, letter in c.greens.items():
            mask &= codes[:, slot] == ord(letter) - 97
        for slot, letters in c.banned.items():
            if letters:
                mask &= ~np.isin(codes[:, slot], [ord(x) - 97 for x in letters])
        for letter, n in c.min_count.items():
            mask &= counts[:, ord(letter) - 97] >= n
        for letter, n in c.max_count.items():
            mask &= counts[:, ord(letter) - 97] <= n
        return mask

    # ----- results ---------------------------------------------------------

    @property
    def remaining_words(self):
        return self.words[self._remaining].tolist()

    @property
    def is_solved(self):
        return bool(self.history) and set(self.history[-1][1]) == {GREEN}

    def display_remaining_words(self, per_line=5):
        """Print the remaining words, ``per_line`` to a line."""
        words = self.remaining_words
        for i in range(0, len(words), per_line):
            print(", ".join(words[i : i + per_line]))
        print(f"number of remaining words is: {len(words)}")

    def remaining_letter_frequency(self, ax=None):
        """Horizontal bar chart of how many remaining words contain each letter,
        leaving out letters already known to be in the answer."""
        import matplotlib.pyplot as plt

        counts = (self._counts[self._remaining] > 0).sum(axis=0)
        known = set(self.constraints.min_count)
        letters = [x for i, x in enumerate(ALPHABET) if counts[i] and x not in known]
        values = [counts[ord(x) - 97] for x in letters]
        if ax is None:
            ax = plt.gca()
        ax.barh(letters[::-1], values[::-1])
        ax.set_xlabel("remaining words containing the letter")
        return ax
