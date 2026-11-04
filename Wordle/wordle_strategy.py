"""Ranking guesses: which word should you play next?

Every method scores each allowed guess against the words that could still
be the answer:

* ``entropy`` (default): the expected information in bits. A guess splits
  the candidates into groups by the colors it would show; the more evenly
  it splits them, the more you expect to learn. 1 bit halves the
  candidates on average, and 2 bits quarter them.
* ``heatmap``: the "heat map method" from ``IDEAS.md``. Each letter scores
  the share of candidates with that letter in that slot (its chance of
  coming back green), plus, once per distinct letter, the share of
  candidates containing it (its chance of coming back green or yellow).
  It needs no lookahead, so it's cheap, but it only approximates entropy.
* ``frequency``: the most common candidate word, by ``word_frequency.csv``.
  This is roughly how a person plays.

Ties are broken in favour of a guess that could itself be the answer,
because it might win outright; then by how common the word is; then
alphabetically.

Colors are encoded as one number per (guess, answer) pair: green = 2,
yellow = 1 and gray = 0 in each slot, read as a base-3 number. So a 5-letter
word has 3**5 = 243 possible patterns, and 242 means all green.
"""
import numpy as np

N_PATTERNS_BASE = 3


def encode(words):
    """Words (equal length, a-z) as an int8 array of letter codes 0-25."""
    words = list(words)
    length = len(words[0]) if words else 0
    return (
        np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8).reshape(len(words), length).astype(np.int8)
        - 97
    )


def letter_counts(codes):
    """Letter counts (rows: words, columns: a-z) from an array of letter codes."""
    counts = np.zeros((len(codes), 26), dtype=np.int8)
    for slot in range(codes.shape[1]):
        np.add.at(counts, (np.arange(len(codes)), codes[:, slot]), 1)
    return counts


def pattern_matrix(guess_codes, answer_codes, chunk=512):
    """The color pattern of every guess against every answer, as a
    ``(len(guesses), len(answers))`` uint8 array of base-3 codes.

    This applies the same rules as ``wordle_solver.score``, vectorized:
    greens first, then yellows from left to right, each using up one
    unmatched copy of the letter in the answer.
    """
    n_guesses, length = guess_codes.shape
    out = np.empty((n_guesses, len(answer_codes)), dtype=np.uint8)
    answers = answer_codes[None, :, :]  # 1 x A x L
    place = N_PATTERNS_BASE ** np.arange(length)
    for start in range(0, n_guesses, chunk):
        g = guess_codes[start : start + chunk][:, None, :]  # c x 1 x L
        green = g == answers  # c x A x L
        unmatched = np.where(green, -1, answers)  # answer letters not used by a green
        code = np.zeros(green.shape[:2], dtype=np.int16)
        yellows = []
        for i in range(length):
            letter = g[:, :, i : i + 1]  # c x 1 x 1
            available = (unmatched == letter).sum(axis=2)  # copies left in the answer
            used = np.zeros_like(available)
            for k in range(i):  # earlier yellows of the same letter use copies first
                used += yellows[k] & (g[:, :, k] == g[:, :, i])
            yellow = ~green[:, :, i] & (available > used)
            yellows.append(yellow)
            code += place[i] * (2 * green[:, :, i] + yellow)
        out[start : start + chunk] = code
    return out


def entropy(patterns, chunk=1024):
    """Expected information (bits) of each guess (row) over equally likely
    candidates (columns). Uses H = log2(m) - (1/m) * sum(c * log2(c)) over
    the sizes c of the groups the guess splits the m candidates into."""
    n_guesses, m = patterns.shape
    if m <= 1:
        return np.zeros(n_guesses)
    n_codes = int(patterns.max()) + 1
    out = np.empty(n_guesses)
    for start in range(0, n_guesses, chunk):
        p = patterns[start : start + chunk].astype(np.int64)
        rows = np.arange(len(p))[:, None] * n_codes
        sizes = np.bincount((rows + p).ravel(), minlength=len(p) * n_codes).reshape(len(p), n_codes)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(sizes > 0, sizes * np.log2(np.maximum(sizes, 1)), 0.0)
        out[start : start + chunk] = np.log2(m) - term.sum(axis=1) / m
    return out


def expected_remaining(patterns, chunk=1024):
    """Expected number of candidates left after each guess (row): the sum of
    squared group sizes divided by the number of candidates. Lower is better."""
    n_guesses, m = patterns.shape
    n_codes = int(patterns.max()) + 1 if patterns.size else 1
    out = np.empty(n_guesses)
    for start in range(0, n_guesses, chunk):
        p = patterns[start : start + chunk].astype(np.int64)
        rows = np.arange(len(p))[:, None] * n_codes
        sizes = np.bincount((rows + p).ravel(), minlength=len(p) * n_codes).reshape(len(p), n_codes)
        out[start : start + chunk] = (sizes.astype(np.float64) ** 2).sum(axis=1) / max(m, 1)
    return out


def heatmap_scores(guess_codes, candidate_codes, candidate_counts):
    """The heat-map score of each guess: per slot, the share of candidates
    with the guess's letter there, plus, per distinct letter, the share of
    candidates containing it."""
    m, length = candidate_codes.shape
    heat = np.zeros((length, 26))
    for slot in range(length):
        heat[slot] = np.bincount(candidate_codes[:, slot], minlength=26) / m
    presence = (candidate_counts > 0).mean(axis=0)  # 26
    positional = heat[np.arange(length)[None, :], guess_codes].sum(axis=1)
    distinct = np.zeros((len(guess_codes), 26), dtype=bool)
    distinct[np.arange(len(guess_codes))[:, None], guess_codes] = True
    return positional + distinct @ presence


def rank(scores, words, is_candidate, frequency, n):
    """Indices of the top ``n`` guesses: by score (rounded so float noise
    doesn't break ties), then candidate first, then frequency, then word."""
    order = np.lexsort((words, -frequency, ~is_candidate, -np.round(scores, 9)))
    return order[:n]
