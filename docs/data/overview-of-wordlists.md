# Wordlists and solution splits

The project keeps its model-facing Wordle vocabularies in [`data/`](../../data/):

- [`wordlist-action-space-4739.csv`](../../data/wordlist-action-space-4739.csv) contains the proposed 4,739-action vocabulary for the model. It includes every valid solution plus additional useful guesses.
- [`wordlist-valid-solutions-all-2309.csv`](../../data/wordlist-valid-solutions-all-2309.csv) is the untouched backup of all 2,309 valid solutions.
- [`wordlist-valid-solutions-train-2109.csv`](../../data/wordlist-valid-solutions-train-2109.csv) contains the 2,109 solutions used to fit model weights.
- [`wordlist-valid-solutions-validation-100.csv`](../../data/wordlist-valid-solutions-validation-100.csv) contains 100 solutions used for evaluation during model and hyperparameter development.
- [`wordlist-valid-solutions-test-100.csv`](../../data/wordlist-valid-solutions-test-100.csv) contains the final 100-solution holdout. Do not use it to make modelling decisions.

Each filename records its word count immediately before the `.csv` extension. The files contain one uppercase,
five-letter word per line, in alphabetical order, with no header. Despite the `.csv` extension, each row has only one
field.

The action space is the union of all 2,309 solutions and 2,430 additional words selected using the
[`subtlex-word-frequencies`](https://github.com/words/subtlex-word-frequencies) distribution. Its SUBTLEX-US counts come
from a 51-million-word corpus of American film subtitles, so the selection favours words found in spoken dialogue and
inherits that corpus's biases. The exact historic frequency cutoff is not recorded in the wordlists repository.

## Provenance

The all-solutions backup and action space are exact copies from the neighbouring `../wordle-ml_wordlists/` repository at
commit `3f32b424c813d22bb2d73e8802e41b78e7d9ba68` (2026-06-08). That repository remains the source of truth for
curating the vocabularies; copies and derived splits here make model training and evaluation self-contained.

The larger valid-guesses list is deliberately not copied. It describes everything accepted by the game, whereas the
model's output indices are defined by the smaller proposed action space.
