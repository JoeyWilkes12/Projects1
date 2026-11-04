# Wordle Solver: Constraint Propagation over a Feature-Engineered Dictionary

> Type in your guess and the colors Wordle gave back. The solver narrows thousands of candidate words down as clues arrive; in the games below, it was down to a handful within two or three guesses.

| | |
|---|---|
| **Concepts** | Constraint satisfaction, information theory (entropy), vectorized filtering, feature engineering, data cleaning, property-based testing, letter-frequency heuristics |
| **Stack** | NumPy, pandas, Matplotlib, pytest |
| **Code** | [`wordle_solver.py`](wordle_solver.py) (v1, current) and [`wordle_engine.py`](wordle_engine.py) (game engine and self-play) and [`wordle_strategy.py`](wordle_strategy.py) (guess ranking), with tests in [`test_wordle_solver.py`](test_wordle_solver.py) and [`test_wordle_engine.py`](test_wordle_engine.py); [`Wordle_Solver_v0.py`](Wordle_Solver_v0.py) (the first version, kept for comparison); three curated word-list CSVs |
| **Notebooks** | [`01_Word_Elimination_v3.ipynb`](01_Word_Elimination_v3.ipynb) (the solver's prototype) and [`wordle_engine.ipynb`](wordle_engine.ipynb) (a playable game that scores guesses), plus three earlier notebooks on building and exploring the datasets |
| **Ideas** | [`IDEAS.md`](IDEAS.md): the original improvement notes, with the status of each |
| **Status** | v1: exact filtering and next-guess suggestions. A full benchmark is in progress (see [`IDEAS.md`](IDEAS.md)) |

---

## Why it's cool

Every Wordle clue is a **constraint**:
- 🟩 *this letter is in this slot*;
- 🟨 *this letter is in the word, but not in this slot*;
- ⬜ *this letter isn't in the word*.

The solver turns each clue into a filter over a table of words and **intersects** the results across all guesses so far. Whatever survives is consistent with every clue.

The trick is in the data. Each word is pre-processed into columns: the letter in each position, plus a yes/no column for each letter of the alphabet. That makes every clue a fast pandas filter instead of string matching in a loop.

## Where it's used in the real world

The same pattern, "turn rules into filters and intersect the results", powers:
- **Search engines and e-commerce:** faceted search, such as "size M, in stock, under $50".
- **Scheduling and logistics:** constraint solvers for timetables, delivery routes, and staff rosters.
- **Security:** narrowing password or key candidates from partial information.
- **Bioinformatics:** filtering sequence candidates against known motifs.
- **Puzzle solvers:** Sudoku, crosswords, and Mastermind (Wordle's ancestor).

The datasets also show a core data-science skill: **feature engineering**, which reshapes raw text into columns a model, or a filter, can use directly.

## The concept in 60 seconds

1. **Represent each word as features:** the letter in each of positions 1 to 5, and a yes/no flag for each letter a to z.
2. **Track what you know:**
   - letters known to be in the word;
   - letters known *not* to be in the word;
   - for each slot, the green letter (if known) and the yellow letters (which are excluded from that slot).
3. **Filter:** keep only the rows that satisfy every rule, by intersecting row indices across all constraints.
4. **Handle repeated letters with counts.** Each guess gives a *minimum* count for every letter that came back green or yellow. If another copy of the letter came back gray, it also gives a *maximum*. For example, `sheep` against `skate` comes back `s!he?ep`: one `e` is yellow and the other is gray, so the answer has exactly one `e`, and it isn't in slot 3 or 4. v0 stored only whether each letter appears, so it couldn't use this; v1 counts letters.
5. **Advise:** chart how often each letter appears among the remaining candidates to help choose the next guess.

## What's implemented

[`wordle_solver.py`](wordle_solver.py) (v1):

| Component | What it does |
|---|---|
| `score(guess, answer)` | Wordle's colors for a guess as a string of `g`, `y` and `b`, with the repeated-letter rules |
| `parse_feedback("wea!r!y")` / `to_notation(...)` | Convert between the typed notation and `("weary", "bbggb")`. Badly formed input raises `ValueError`. |
| `Constraints` | What's known so far: the green letter in each slot, letters banned from each slot, and a minimum and maximum count for each letter. Feedback that contradicts an earlier guess raises `ValueError`. |
| `WordleSolver(words=None)` | Loads a word list (a CSV path, a DataFrame, or a list of words; the NLTK CSV by default) and stores each word as NumPy arrays of letter codes and letter counts |
| `add_guess("wea!r!y")` | Adds one guess and re-checks only the words still remaining |
| `solve_from_history([...])` | Applies a whole game's feedback, then filters the word list once |
| `remaining_letter_frequency()` / `display_remaining_words()` | The letter-frequency chart and printed list from v0 |
| `letter_count_table(words)` | One row per word, one column per letter, holding how many times the letter appears |

**How it's tested** (`python -m pytest Wordle`, 64 tests in about 20 seconds):
- In 250 simulated games, the remaining words are compared with a brute-force search: every word that would have produced exactly the same colors. They match every time; 100 of the games use guesses with repeated letters.
- Four real games from v0's `test_class()` are replayed, and the solver must end with exactly the answer.
- There are also tests for scoring edge cases, notation errors, and contradictory feedback.
- The engine is checked against five real games, including the `butch` game from the notebook. Add your own to `REAL_GAMES` in `test_wordle_engine.py`.
- Self-play runs through all 3,624 madalynrose words. Each game asserts that the solver never dropped the answer.
- The pattern table is checked against `score()`, including repeated letters.
- Suggestion tests cover entropy and heat-map scores on small examples, hard mode only suggesting consistent words, and the frequency tie-break. The entropy strategy must win 60 sampled games.

### Suggesting the next guess

`WordleSolver.suggest(n=5, method="entropy", hard_mode=False)` ranks every allowed guess against the words that could still be the answer.

| Method | Score | Idea |
|---|---|---|
| `entropy` (default) | Expected information in bits | A guess splits the candidates into groups by the colors it would show. The more evenly it splits them, the more you expect to learn: 1 bit halves the candidates on average. |
| `heatmap` | Letter-position frequency | The "heat map method" from the MadalynRose notebook. For each slot, the share of candidates with the guess's letter there, plus, for each distinct letter, the share of candidates containing it. Cheap, with no lookahead. |
| `frequency` | Word popularity | The most common candidate word. This is roughly how a person plays. |

Ties go to a guess that could itself be the answer, since it might win outright, then to the more common word ([`word_frequency.csv`](word_frequency.csv)), then alphabetically.

```python
>>> ws = WordleSolver()
>>> ws.suggest(3)
    word     score  candidate  frequency
0  tares  6.358716       True       1.90
1  rates  6.252396       True       4.88
2  tears  6.227391       True       4.47
```

To score guesses fast, the solver builds a table of the color pattern for every (guess, answer) pair: 9,795 × 3,624 patterns, each stored as a base-3 number. It is built with vectorized NumPy on first use, which takes about 7 seconds and 35 MB. The tests check it against `score()`.

**Hard mode and normal mode.** In hard mode (`hard_mode=True`), every guess must be consistent with all the clues so far. That's stricter than the NYT rule, which only requires greens to stay put and yellows to be reused. As the note in [`IDEAS.md`](IDEAS.md) says, hard mode is *easier* for a computer: there are fewer guesses to score. Normal mode lets the solver play a word that can't be the answer but splits the candidates better. After `t!ar?es`, for example, 18 words remain, and the best guess is `choup`, which isn't one of them.

**Sample results**, playing 400 randomly chosen answers from the madalynrose list:

| Strategy | Won in 6 | Average guesses | Worst game |
|---|---|---|---|
| `entropy` | **100%** | **3.57** | 5 |
| `entropy`, hard mode | 98.0% | 3.78 | 9 |
| `heatmap` | 98.3% | 3.86 | 11 |
| `frequency` | 94.8% | 4.42 | 9 |

A full benchmark over every answer, and a ranking of starting words, is the next PR.

### Word lists

By default, `WordleSolver()` uses separate lists for:
- **possible answers:** the madalynrose list, 3,624 common words;
- **allowed guesses:** the joined list, 9,795 words.

The NLTK list includes rare words, such as `urari` and `uzara`, that are unlikely answers but are fine to guess. Pass `WordleSolver(words, guesses)` to use other lists, or `WordleSolver(words)` to guess only from `words`.

`word_frequency.csv` is built by [`build_word_frequency.py`](build_word_frequency.py) from the [`wordfreq`](https://github.com/rspeer/wordfreq) package by Robyn Speer. Its data is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), so `word_frequency.csv` is shared under the same license. The solver only reads the CSV, so `wordfreq` isn't needed to run it.

[`wordle_engine.py`](wordle_engine.py):

| Component | What it does |
|---|---|
| `Wordle(wordle=None, allowed=None, seed=None)` | The game from `wordle_engine.ipynb` as a module, using the solver's `score` function. `guess()` returns feedback in the solver's notation. With no wordle given, it picks a random word. `allowed` restricts guesses to a word list, as the real game does. |
| `Wordle.play(guesses=None)` / `Wordle.from_hidden_input()` | Play interactively or replay a list of guesses; set the word with hidden input |
| `self_play(answer, strategy)` | The solver plays the engine: `strategy(solver)` picks each guess. It raises if the solver ever drops the answer. |
| `random_candidate(seed)` / `best_guess(method, hard_mode)` | Strategies: a random word that could still be the answer (v0's planned `random_solver`), or the solver's top suggestion |
| `self_play(..., first_guess="crane")` / `simulate(..., first_guess=...)` | Fix the opening word |
| `simulate(answers, strategy)` / `summarize(results)` | Play one game per answer; report the win rate, average guesses and the distribution of guesses per game |

**Self-play baseline:** the `random_candidate` strategy, playing every word in each list (games continue past six guesses so the full distribution is visible):

| Word list | Games | Won in 6 | Average guesses | Worst game |
|---|---|---|---|---|
| madalynrose | 3,624 | 94.5% | 4.39 | 12 |
| NLTK | 8,497 | 91.9% | 4.74 | 12 |

The 200 madalynrose losses come from word families that differ by one letter, such as `?ears` (dears, fears, gears, …), `?ated` and `?ales`. Guessing those candidates one at a time runs out of turns. A strategy that picks guesses to split the candidates is the next PR.

[`Wordle_Solver_v0.py`](Wordle_Solver_v0.py) is the first version, kept for comparison. It filters with pandas and stores only whether each letter appears, so with repeated letters it can keep words the clues rule out (see [`IDEAS.md`](IDEAS.md)).

### Input notation
Type each letter of your guess. Follow a letter with `!` if it came back green or `?` if it came back yellow. Leave gray letters bare.

| Wordle shows | You type |
|---|---|
| ⬜⬜🟩🟩⬜ for **WEARY** | `wea!r!y` |
| ⬜⬜⬜⬜🟨 for **LEARN** | `learn?` |

## How it evolved: the notebooks

The solver started as notebook experiments in Google Colab. Two of them are included here because they show the logic before it was packaged into a class.

### [`01_Word_Elimination_v3.ipynb`](01_Word_Elimination_v3.ipynb): the prototype solver

This is the filtering logic from `Wordle_Solver_v0.py`, written by hand before it became a class.
- Each cell holds one game's constraints. You edit the `in_word` and `not_in_word` lists and the per-slot conditions (`df['3rd'] == 'a'` for a green, `~df['2nd'].isin(...)` for a yellow), then intersect the matching row indices.
- There are four copies of the cell, named `top_left`, `top_right`, `bottom_left`, and `bottom_right`. They were used to play four puzzles at once, as in four-board Wordle variants.
- The letter-frequency bar chart of the remaining candidates later became `remaining_letter_frequency()`.
- The last cell is a first try at parsing the `!`/`?` input notation, which became `new_guess_results()`.
- The comments record the limitation that drives the roadmap: storing whether a letter appears, rather than how many times, loses information about double letters.

### [`wordle_engine.ipynb`](wordle_engine.ipynb): the game engine

This notebook is the other side of the problem: a `Wordle` class that holds a secret word and scores guesses against it.
- `getpass` hides the secret word as it's typed, so one person can set a puzzle for another.
- `score_guess` handles repeated letters in two passes. It marks greens first and uses up those letters, then hands out yellows from what's left, left to right. A letter that appears more often in the guess than in the answer is gray for the extra copies.
- The output is written in the solver's input notation: `sauce` against `butch` prints `sau?c!e`. The engine's output can be passed straight to the solver, which is the basis for the planned self-play mode.
- `play_game` allows six guesses and reveals the word if you run out. A typo, such as the wrong length or a digit, prints an error and doesn't use up a guess.
- `play_game(guesses=[...])` replays a list of guesses instead of prompting. The notebook uses it to replay a real game: `sauce`, `hunch`, `mulch`, and `dutch`, then solved with `butch`.
- `test_score_guess()` checks the scoring on the double-letter cases above and confirms that bad input is rejected.

I checked `score_guess` against an independent implementation of Wordle's coloring rules on 3,004 random guess/answer pairs from the NLTK list, plus the double-letter cases in the notebook (`vivid`/`vviiv`, `eerie`/`leaee`). Every result matched.

### Earlier notebooks: building and exploring the datasets

These three are kept as they were last run in Colab. They read from and write to a Google Drive folder, so they don't run as-is from this repo.

| Notebook | What it shows |
|---|---|
| [`Parse_NLTK_words_Corpora_v0.ipynb`](Parse_NLTK_words_Corpora_v0.ipynb) | How the NLTK CSVs were built. It loads the 235,892-word NLTK `words` corpus, drops proper nouns (words starting with a capital), and writes one row per word: the letter in each position, a yes/no column for each letter, and vowel and consonant counts (`y` counts as a vowel). It writes one CSV per word length. |
| [`v1_-_MadalynRose_bank_analysis.ipynb`](v1_-_MadalynRose_bank_analysis.ipynb) | Where each letter tends to sit in a 5-letter word: 26 small charts of letter count by position. This is the "heat map method" in [`IDEAS.md`](IDEAS.md). |
| [`v2_-_MadalynRose_bank_analysis.ipynb`](v2_-_MadalynRose_bank_analysis.ipynb) | The share of words that contain each letter: `s`, `e` and `a` lead, and `j`, `q` and `z` trail. |

Known issues, left as they are because these are historical notes:
- In the parsing notebook, `number_of_consonants` is computed as `5 - number_of_vowels`. That is only right for 5-letter words, and the notebook's own output shows -2 for a 16-letter word. The vowel-count step also indexes a DataFrame with a `set`, which pandas 2 and later reject.
- The `In Word List` column in the NLTK CSV was filled in by hand later, for 17 vowel-heavy starting words.
- The second chart in the v2 notebook uses `letters` and `counts` from a line that is commented out, so it only worked in the original Colab session.

Both of the main notebooks above run top to bottom from the `Wordle/` folder, and their saved outputs come from that run. The interactive game is in a commented-out cell at the end of `wordle_engine.ipynb`, because it waits for typed input.

## Results

Real games, replayed through both versions of the solver using the NLTK dataset (`WordleSolver(load_words())` for v1):

| Guess | Feedback typed | Left (v0) | Left (v1) |
|---|---|---|---|
| WEARY | `wea!r!y` | 63 | 63 |
| APART | `apa!r!t` | 40 | **29** |
| BOARD | `boa!r!d!` | 6 | 6 |
| GUARD | `gua!r!d!` | 5 | 5 |
| **SHARD** | `s!h!a!r!d!` | solved | solved |

In APART the first `a` comes back gray and the second green, so the answer has exactly one `a`, and it's in slot 3. v1 uses this; v0 couldn't.

| Guess | Feedback typed | Left (v0) | Left (v1) |
|---|---|---|---|
| LEARN | `learn?` | 301 | 301 |
| SOUND | `so!u!n!d` | 4 (count, fount, mount, young) | 4 |
| TICKS | `t?icks` | 2 (fount, mount) | 2 |
| **MOUNT** | `m!o!u!n!t!` | solved | solved |

v1 takes about 0.3 ms per guess on the 8,497-word list.

## Datasets

| File | Rows | Columns |
|---|---|---|
| `5 letter words parsed - NLTK Words Corpora - v0.csv` | 8,497 | word, the letter in each of the 5 positions, one yes/no column per letter a–z, and vowel and consonant counts (unique and total) |
| `madalynrose list of 5 letter words parsed v0.csv` | 3,624 | a community 5-letter word list, with similar position and letter columns |
| `Joined NLTK & Github 5 letter words v0.csv` | 9,795 | the two lists combined and deduplicated |

`WordleSolver()` uses the madalynrose list as possible answers and the joined list as allowed guesses (see **Word lists** above). `load_words()` on its own reads the NLTK file. A larger answer list catches rare answers but leaves more candidates per clue.

One data fix happens at load time: the NLTK CSV stores the word `false` as `FALSE`, because a spreadsheet read it as a boolean. `load_words` lowercases every word.

## Run it

```bash
cd Wordle
python
```

```python
>>> from wordle_solver import WordleSolver
>>> ws = WordleSolver()                         # madalynrose answers, joined-list guesses
>>> len(ws.add_guess("learn?"))
172
>>> ws.add_guess("so!u!n!d")
['count', 'mount', 'young']
>>> ws.suggest(1)                               # what to play next
    word     score  candidate  frequency
0  count  1.584963       True       4.83
>>> ws.solve_from_history(["learn?", "so!u!n!d", "t?icks"])   # a whole game at once
['mount']
```

Run the tests from the repository root:

```bash
python -m pytest Wordle
```

To run the notebooks, start Jupyter from this folder so the word list loads from disk:

```bash
cd Wordle
jupyter notebook wordle_engine.ipynb
```

## Skills demonstrated

- Turning an ambiguous, real-world problem into precise constraints.
- Data engineering: building, cleaning, joining, and feature-engineering text datasets.
- Vectorized NumPy filtering, and pandas for loading and reshaping the data.
- Testing a solver against a brute-force reference on randomly generated games.
- Iterative product development: validated against real games, with a clear roadmap.

## Roadmap

The original planning notes, the status of each item, and a measurement of how tight the solver's filtering is are in [`IDEAS.md`](IDEAS.md).

- [x] **Letter counts:** use how many times each letter appears, not just whether it does, to handle double letters fully.
- [x] **Contradiction detection:** badly formed or contradictory feedback raises an error that says what's wrong.
- [x] **Self-play mode:** the engine picks or is given a hidden word, the solver plays it, and `simulate` reports average guesses over a word list.
- [x] **Best-guess suggestions:** rank the next guess by expected information gain (entropy), the heat-map score, or word frequency.
- [ ] **Benchmark and best starting word:** play every answer and rank opening words.

## Further reading

- [Wordle (Wikipedia)](https://en.wikipedia.org/wiki/Wordle)
- [Constraint satisfaction problem](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem)
- [NLTK corpora](https://www.nltk.org/nltk_data/)
