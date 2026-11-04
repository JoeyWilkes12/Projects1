# Unit Testing with pytest: Finding the Bugs Before Users Do

> Code you haven't tested is code you're hoping works. This project uses `pytest` to find and fix real bugs in a set of functions, from prime factors to the card game **SET**.

| | |
|---|---|
| **Concepts** | Unit testing, edge cases, test fixtures, exception testing, input validation, operator overloading |
| **Stack** | Python, pytest |
| **Code** | [`specs.py`](specs.py) (code under test), [`test-specs.py`](test-specs.py) (test suite) |

---

## Why it's cool

The functions in `specs.py` look correct, but `smallest_factor` and `Fraction` started out with bugs, as the comments in the test file note ("write a unit test …, then correct it"). The approach:
1. Write tests aimed at the inputs most likely to break the code, such as primes, perfect squares, February in a leap year, and a zero denominator.
2. Watch them fail.
3. Fix the code.

The suite now guards against those bugs coming back.

There's a bit of math in it too. Checking whether three SET cards form a valid "set" reduces to one line of **modular arithmetic**.

## Where it's used in the real world

- **Everywhere software ships:** continuous integration (GitHub Actions, Jenkins) runs suites like this on every commit, and a failing test blocks a bad merge.
- **Refactoring safely:** tests let engineers rewrite code knowing behavior hasn't changed.
- **Test-driven development (TDD):** write the failing test first, then the code that makes it pass.
- **Regulated industries:** finance, medical devices, and aerospace require documented test coverage.
- **Open source:** contributors' changes are judged by whether the existing tests still pass.

## The concept in 60 seconds

- **Unit test:** a small function that calls your code with known inputs and `assert`s the expected output.
- **Edge cases:** the inputs most likely to break code, such as boundaries (0, 1, empty), special values (primes, perfect squares, leap years), and invalid inputs.
- **Testing errors:** `with pytest.raises(ValueError):` checks that bad input fails *loudly* and with the right message.
- **Fixtures:** `@pytest.fixture` builds shared test data once and hands it to every test that needs it.
- **Running:** `pytest` finds and runs every `test_*` function and reports what failed and why.

## What's implemented

### Code under test ([`specs.py`](specs.py))

| Function / class | Purpose | Edge cases tested |
|---|---|---|
| `smallest_factor(n)` | Smallest prime factor | Primes, perfect squares (16), products of two primes (221 = 13 × 17) |
| `month_length(month, leap_year)` | Days in a month | Every month, February with and without a leap year, invalid month names |
| `operate(a, b, oper)` | Arithmetic dispatcher | Each operator, non-string operator (`TypeError`), unknown operator (`ValueError`), divide by zero |
| `Fraction` | A reduced fraction with `+ - * /`, `==`, `str`, and `float` | Reducing to lowest terms, zero denominator, non-integer input, comparing with floats |
| `count_sets(cards)`, `is_set(a, b, c)` | Counts valid "sets" in a 12-card hand of the game SET | Wrong number of cards, duplicate cards, malformed cards, non-base-3 digits, a known hand with exactly 6 sets |

### The SET trick
Each SET card has 4 attributes, each with 3 values, so a card is a 4-digit base-3 string like `"1022"`. Three cards form a set when every attribute is **all the same or all different**. For digits in {0, 1, 2}, that's true exactly when their **sum is divisible by 3**:

```python
for i in range(4):
    if (int(a[i]) + int(b[i]) + int(c[i])) % 3 != 0:
        return False
return True
```

## Results

```bash
$ python -m pytest test-specs.py
...............                                                   [100%]
15 passed
```

## Run it

```bash
cd Unit_testing
python -m pytest test-specs.py -v
```

The file name `test-specs.py` uses a hyphen, so `pytest` won't find it automatically; pass the file name explicitly as shown. Renaming it to `test_specs.py` would let a bare `pytest` find it.

## Skills demonstrated

- Professional testing practice: edge-case analysis, testing for exceptions, fixtures, and clear assertion messages.
- Defensive programming: input validation with specific, testable error messages.
- Python's data model: building a numeric type with `__add__`, `__eq__`, `__float__`, and the other operator methods.
- Spotting a mathematical shortcut (modular arithmetic) that turns a fiddly check into one line.

## Known gaps (a good exercise)

- `count_sets` checks for duplicate cards with `a is b`, which tests *object identity*, not equality. The test passes because Python reuses identical string literals, but a duplicate card built at runtime (for example, read from a file) slips through. Write a test that exposes this, then fix it with `len(set(cards)) != len(cards)`.
- The "exactly 4 digits" check only looks at the *total* length of all the cards. A hand with one 3-digit card and one 5-digit card passes that check, then crashes with `IndexError` instead of raising the documented `ValueError`. Add a test for that case too.
- Measure coverage with `pytest --cov` to find untested lines.

## Further reading

- [pytest documentation](https://docs.pytest.org/)
- [Unit testing (Wikipedia)](https://en.wikipedia.org/wiki/Unit_testing)
- [SET (card game)](https://en.wikipedia.org/wiki/Set_(card_game))
