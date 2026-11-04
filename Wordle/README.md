# Wordle Solver: Constraint Propagation over a Feature-Engineered Dictionary

> Type in your guess and the colors Wordle gave back. The solver narrows thousands of candidate words down as clues arrive; in the games below, it was down to a handful within two or three guesses.

| | |
|---|---|
| **Concepts** | Constraint satisfaction, set intersection, feature engineering, data cleaning, letter-frequency heuristics |
| **Stack** | pandas, NumPy, Matplotlib |
| **Code** | [`Wordle_Solver_v0.py`](Wordle_Solver_v0.py), plus three curated word-list CSVs |
| **Status** | v0, in active development |

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
4. **Handle repeated letters:** if a guess has two `a`'s and one comes back gray while the other is green, `a` is still in the word. The solver removes confirmed letters from the "not in word" set so they aren't wrongly excluded.
5. **Advise:** chart how often each letter appears among the remaining candidates to help choose the next guess.

## What's implemented

| Component | What it does |
|---|---|
| `Wordle_Solver(wordle_length=5, ...)` | Loads a word dataset (a pandas DataFrame or CSV path) and sets up the constraint state. Word length is configurable. |
| `new_guess_results(guess_results)` | Parses one guess with its feedback, updates the constraints, filters the candidates, and returns the remaining words; detects when the puzzle is solved |
| `remaining_letter_frequency()` | Bar chart of letter frequency across the remaining candidates, leaving out letters already known |
| `display_remaining_words()` | Prints the remaining candidates, five per line |

### Input notation
Type each letter of your guess. Follow a letter with `!` if it came back green or `?` if it came back yellow. Leave gray letters bare.

| Wordle shows | You type |
|---|---|
| ⬜⬜🟩🟩⬜ for **WEARY** | `wea!r!y` |
| ⬜⬜⬜⬜🟨 for **LEARN** | `learn?` |

## Results

Real games, replayed through the solver using the NLTK dataset:

| Guess | Feedback typed | Candidates left |
|---|---|---|
| WEARY | `wea!r!y` | 63 |
| APART | `apa!r!t` | 40 |
| BOARD | `boa!r!d!` | 6 |
| GUARD | `gua!r!d!` | 5 |
| **SHARD** | `s!h!a!r!d!` | **solved** |

| Guess | Feedback typed | Candidates left |
|---|---|---|
| LEARN | `learn?` | 301 |
| SOUND | `so!u!n!d` | 4 (count, fount, mount, young) |
| TICKS | `t?icks` | 2 (fount, mount) |
| **MOUNT** | `m!o!u!n!t!` | **solved** |

## Datasets

| File | Rows | Columns |
|---|---|---|
| `5 letter words parsed - NLTK Words Corpora - v0.csv` | 8,497 | word, the letter in each of the 5 positions, one yes/no column per letter a–z, and vowel and consonant counts (unique and total) |
| `madalynrose list of 5 letter words parsed v0.csv` | 3,624 | a community 5-letter word list, with similar position and letter columns |
| `Joined NLTK & Github 5 letter words v0.csv` | 9,795 | the two lists combined and deduplicated |

The solver uses the NLTK file by default. A larger dictionary catches rare answers but leaves more candidates per clue.

## Run it

```bash
cd Wordle
python
```

```python
>>> from Wordle_Solver_v0 import Wordle_Solver
>>> ws = Wordle_Solver(wordle_length=5)        # loads the NLTK CSV in this folder
>>> ws.new_guess_results("learn?")             # 301 candidates + letter-frequency chart
>>> ws.new_guess_results("so!u!n!d")           # ['fount', 'count', 'young', 'mount']
```

Pass `display_remaining_letter_frequency=False` to skip the chart.

## Skills demonstrated

- Turning an ambiguous, real-world problem into precise constraints.
- Data engineering: building, cleaning, joining, and feature-engineering text datasets.
- pandas fluency: boolean masks, index-set intersection, and `isin` filters.
- Iterative product development: validated against real games, with a clear roadmap.

## Roadmap

- [ ] **Letter counts:** store how many times each letter appears, not just whether it does, to handle double letters fully.
- [ ] **Contradiction detection:** warn when typed feedback is inconsistent, for example after a typo.
- [ ] **Self-play mode:** the solver picks a hidden word, plays itself, and reports average guesses over the dictionary.
- [ ] **Best-guess suggestions:** rank the next guess by expected information gain (entropy).

## Further reading

- [Wordle (Wikipedia)](https://en.wikipedia.org/wiki/Wordle)
- [Constraint satisfaction problem](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem)
- [NLTK corpora](https://www.nltk.org/nltk_data/)
