"""Build word_frequency.csv: how common each 5-letter word is in English.

    pip install wordfreq
    python build_word_frequency.py

Writes one row per word in the joined NLTK + madalynrose list, with the
word's Zipf frequency from the ``wordfreq`` package: log10 of how many
times it appears per billion words. 3 is about once per million words, 6
about once per thousand, and 0 means wordfreq hasn't seen it.

The solver uses this column only to break ties between equally good
guesses (the "word popularity as a tie breaker" item in IDEAS.md), so it
doesn't need wordfreq installed to run.

Attribution: the frequencies come from wordfreq by Robyn Speer
(https://github.com/rspeer/wordfreq), whose data is licensed under
CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). The
generated word_frequency.csv is shared under the same license.
"""
import os

import pandas as pd
from wordfreq import zipf_frequency

from wordle_solver import GUESSES_CSV, HERE, load_words

OUT = os.path.join(HERE, "word_frequency.csv")


def main():
    words = load_words(GUESSES_CSV)
    table = pd.DataFrame({"word": words, "zipf": [zipf_frequency(w, "en") for w in words]})
    table.to_csv(OUT, index=False)
    print(f"wrote {len(table)} words to {OUT}")
    print(table.sort_values("zipf", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
