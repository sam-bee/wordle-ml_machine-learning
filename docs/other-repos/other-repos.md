# Other Repos

There are some other repos which are important. Broadly, they are previous steps in this project. Some concern training
data processing, for example. Their docs and version control histories are to be treated with a little caution however -
the approach may have changed since they were written.

From the root directory of this repository, see:
- `../wordle-ml_wordlists/`
- `../wordle-ml_game-engine/`
- `../wordle-ml_synthetic-data-creation/`

## Wordlists

These are wordlists that provide a set of legal actions in the game of Wordle, legal solutions to a Wordle, and a curated action space.

## Game Engine

The authoritative Wordle feedback implementation. Synthetic-data release
`v0.1.0` and this module use game-engine release `v1.0.1`, which correctly
handles repeated letters by consuming matched occurrences.

## Synthetic Data Creation

A Wordle teacher and offline corpus generator. Release `v0.1.0` owns the WDIT
v3 artifacts copied into `data/imitation/` and is imported here for its reader
and record contract. Expensive teacher ranking remains in that repository;
this repository expands the frozen records into model inputs on demand.
