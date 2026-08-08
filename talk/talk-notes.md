# Talk notes

## Where the data came from

- The solution and accepted-guess lists originated in the dictionary shipped in Wordle's browser code. The NYT-era
  snapshot used here has 2,309 possible solutions and 12,947 accepted guesses.
- The proposed action space combines every solution with 2,430 additional five-letter words selected using SUBTLEX-US
  frequencies, derived from 51 million words of American film subtitles. That favours spoken vocabulary, but also
  carries the corpus's names, slang, and other biases into the model.
- The lists were normalised to uppercase ASCII, de-duplicated, alphabetically sorted, and stored one word per row. The
  result is 4,739 fixed model outputs; the exact historic frequency cutoff was not recorded, a useful reproducibility
  lesson in its own right.

The talk is one hour for Go developers who may have little machine-learning experience. It must build the ML concepts
gradually and keep the Wordle problem visible throughout, rather than turning into a tour of infrastructure.

## Reproducible starting point

- Begin demonstrations from the same containerised environment used during development.
- Show that GPU selection is a real engineering concern: the development desktop has two NVIDIA cards, but Compose
  passes only the RTX 5070 Ti by UUID.
- Use the small CUDA smoke kernel to make `sm_120`, compute capability 12.0, host/device memory, and kernel launch syntax
  concrete before introducing model code.
- Follow it with the GoMLX Euclidean-distance graph to introduce symbolic graphs, XLA compilation, and the CUDA backend
  without pretending that the Wordle model already exists.
- Mention TensorBoard and the visualiser only briefly at this stage. Their deliberately empty states make a useful
  before-and-after contrast once training metrics and gameplay data are implemented.

## Wordle as a model problem

- Contrast the 2,309 possible solutions with the proposed 4,739-action vocabulary. A legal answer and a useful guess
  are related but distinct concepts.
- Use the fixed action vocabulary to make the model output concrete: one score per possible action, with a stable word
  at each output index.
- Keep the larger set of all game-legal guesses outside the initial model. This simplifies the demo while retaining
  every possible solution as an action.
- Split by solution, not by generated game state: 2,109 answers for training, 100 for validation while tuning, and 100
  held back for one final test. This prevents states for the same answer leaking across datasets.
- Make the split deterministic and keep the full 2,309-word backup. Repeatedly consulting the final test set—or trying
  new splits after seeing scores—turns human judgement into another way of overfitting.

## Later structure

[To be expanded alongside the model: Wordle data and encoding, first baseline, loss and optimisation, Go/CUDA boundary,
training results, failures and lessons, live visualisation, and conclusion.]
