# Wordle: ideas and improvement notes

These are my original planning notes for the Wordle solver, followed by where each item stands now. The status column reflects the code in this folder, and the measurements come from running it.

## Original notes

> Note the irony that hard mode is easy to a computer but harder for a person, whereas not hard mode is harder to a computer because there are more possibilities and greater potential payout.
>
> Improvements
>
> * finish solver
>    * Create metrics for remaining words to select best word
>    * Add into dataset word popularity as a tie breaker
>    * Add in parameter to choose first word guess
> * Improve solver temporal complexity
>    * When given a list of guesses, the solver shouldn't have to recreate the list of remaining words after each guess but parse the list of guesses and results then identify list of remaining words
>       * This shouldn't be a problem when connected to the game engine
> * Remaining words improvement
>    * Right now the solver Durant take advantage of information around duplicate letters
>    * Need to update dataset so that it's not just a boolean for each letter : as to in wordle or not, but that it's the number of times that letter appears in that word
> * Incorporate heat map method
>
> Identify best starting word
>
> * Scale up testing of engine and solver

## Status

| Item | Status | Where things stand |
|---|---|---|
| **Finish the solver** | Partly done | `Wordle_Solver.new_guess_results` filters the candidates correctly. It doesn't yet suggest a next guess, and `random_solver()` is an empty stub. |
| Metrics to pick the best word | Not started | The only aid is the letter-frequency chart (`remaining_letter_frequency`), which a person has to read. |
| Word popularity as a tie-breaker | Not started | None of the three datasets has a frequency column. The parsing notebook has a commented-out `nltk.download('brown')`; the Brown corpus is one possible source of word counts. |
| Parameter for the first guess | Not started | |
| **Faster solving from a list of guesses** | Not started | `new_guess_results` takes one guess at a time. Each call re-filters the full table for every constraint so far. Speed is not the problem: one guess takes about 20 ms on the 8,497-word NLTK list. The gain would be a cleaner API for replaying a game in one call. |
| **Use repeated-letter information** | Partly done | The solver avoids the worst mistake. When a guess has two `a`s and one comes back gray, `a` is not excluded from the word. The datasets still store a yes/no flag per letter instead of a count, so some information is still lost (see the measurement below). |
| Store letter counts in the dataset | Not started | The `a`–`z` columns are booleans in all three CSVs. |
| **Heat map method** | Explored, not in the solver | `v1_-_MadalynRose_bank_analysis.ipynb` plots how often each letter appears in each of the five positions. `Wordle_Solver_v0.py` imports seaborn "for heatmap" but doesn't use it. |
| **Best starting word** | Groundwork only | The `In Word List` column of the NLTK CSV records a hand check of 17 vowel-heavy openers (`adieu`, `audio`, `ourie`, …) against the list of words Wordle accepts. No ranking has been done. |
| **Scale up testing** | Partly done | Engine: `test_score_guess()` in `wordle_engine.ipynb` asserts the double-letter cases and the bad-input checks. Solver: `test_class()` lists several real games, but each one overwrites the last, so only the `shard` game runs. It has no asserts, and its call is commented out. |

### Measurement: how tight is the solver?

I simulated games with the solver, scoring each guess with Wordle's rules, and compared its remaining words with the exact set of words consistent with every clue.

- **It never drops the answer.** In 400 games of three random guesses each, the true answer was always among the remaining words.
- **It sometimes keeps words it should have dropped.** In 48 of 300 games (16%), after two guesses it kept words that the clues ruled out. On average it kept 126 candidates where 118 were possible.

The extra words come from information the solver doesn't record:
- how many copies of a letter the word has, for example "exactly one `e`";
- which slot a gray duplicate letter can't be in.

Both need letter counts.

## Other ideas found in the notebooks

| Idea | Source | Status |
|---|---|---|
| Remove proper nouns from the NLTK corpus | `Parse_NLTK_words_Corpora_v0.ipynb` | Done. Words starting with a capital letter are skipped, which takes the corpus from 235,892 words to 210,693. |
| Compare the madalynrose word list with the NLTK list | `Parse_NLTK_words_Corpora_v0.ipynb` | Done. 1,299 of the 3,624 madalynrose words aren't in NLTK; many are inflected forms such as `wiped`. That is why `Joined NLTK & Github 5 letter words v0.csv` (9,795 words) exists. |
| `fewer` isn't in the NLTK list; "maybe join the two datasets" | `01_Word_Elimination_v3.ipynb` | Done (the joined CSV), though the solver still loads the NLTK list by default. |
| A 6-letter word bank, or a bigger 5-letter one | both MadalynRose notebooks | Partly done. The parsing notebook writes a CSV for any word length, and the solver takes a `wordle_length` argument. Only the 5-letter CSVs are in the repo. |
| Number of duplicates of each letter, per word | `v2_-_MadalynRose_bank_analysis.ipynb` | Not started. This is the same work as "store letter counts". |
| Separate likely answers from words Wordle merely accepts | `01_Word_Elimination_v3.ipynb` (comments on `cothy`, `zloty`) | Not started. The NLTK list includes rare words such as `urari` and `uzara`, which make the candidate lists longer than they need to be. |
| Validate the `!`/`?` input notation | `01_Word_Elimination_v3.ipynb` (last cell) | Not started. It overlaps with "contradiction detection" in the README roadmap. |
| Choose a random word as the wordle | `wordle_engine.ipynb` | Not started. Self-play needs it. |
| Hard mode | the note above | Not started. Neither the engine nor the solver has a hard-mode option. |
