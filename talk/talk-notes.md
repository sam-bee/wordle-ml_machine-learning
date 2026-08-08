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
  before introducing the full Wordle policy graph.
- Use the small standard-library TensorBoard writer as a reassuring Go detail: it writes ordinary scalar event files,
  not a new monitoring system. There are no learning curves yet; those begin with the first experiment.

## Wordle as a model problem

- Contrast the 2,309 possible solutions with the proposed 4,739-action vocabulary. A legal answer and a useful guess
  are related but distinct concepts.
- Use the fixed action vocabulary to make the model output concrete: one score per possible action, with a stable word
  at each output index.
- Keep the larger set of all game-legal guesses outside the initial model. This simplifies the demo while retaining
  every possible solution as an action.
- Split by solution, not by generated game state: 2,109 answers for training, 100 for validation while tuning, and 100
  held back for one final test. This prevents states for the same answer leaking across datasets.
- Keep the full 2,309-word backup alongside the fixed split. Repeatedly consulting the final test set turns human
  judgement into another way of overfitting.
- Freeze the generated examples as a versioned offline corpus: WDIT v3 release `v0.1.0` contains 52,726 training
  records, 1,600 mini records, and 2,500 records in each validation and final-test split. Teacher ranking is not part of
  the training hot path.

## A compact policy model

- Start with the four model-facing tensors rather than raw coloured tiles: the remaining-solution mask, 209 aggregate
  candidate statistics, a turn index, and a mask showing which actions are still possible solutions.
- Show the tiny shared encoder boundary: a 289-byte LSB-first remaining-solution bitset and one turn become those four
  tensors. The same code serves generated training records and later live play, so the demo cannot quietly train on one
  representation and play with another.
- Freeze word IDs with the five checked-in lists and their normalized SHA-256 hashes. This makes a dataset's "word 17"
  verifiable instead of an accidental consequence of whichever dictionary happened to be loaded.
- Normalize the 2,309-value candidate mask by its row sum. Its 96-value linear projection can then be explained as a
  learned mean over the remaining candidates; the separate log candidate-count statistic restores information about
  whether that set contains two words or two thousand.
- Project the statistics to 48 values and embed the six possible turns into 16 values. Concatenating `96 + 48 + 16`
  produces a 160-value state without needing a learned skip-path projection.
- Use one two-layer residual block at width 160. This is enough structure to introduce activations, hidden features,
  residual connections, and trainable weights without making a one-hour Go talk about an oversized architecture.
- Produce 4,739 ordinary action logits, then learn one scalar candidate bonus per state. Adding that scalar only at
  actions which remain solutions gives the network an explicit exploit-versus-probe control.
- Emphasise that the candidate-action mask is not a hard-mode or legality mask. A probe word retains its ordinary logit;
  it is never replaced with negative infinity merely because it cannot be the answer.
- For imitation learning, keep a separate availability mask: it starts at one
  for every action and only clears prior history guesses. This makes the loss
  prevent duplicate guesses without accidentally teaching the candidate bonus
  input to behave like a legality rule.
- Count the weights in front of the audience: `96S + 161A + 61,953`. With the actual vocabularies this is 1,046,596
  FP32 parameters, just under 4 MiB of weights. The output layer accounts for most of them, which makes the fixed action
  vocabulary an architectural decision rather than just a data detail.
- Keep softmax, loss, optimisation, teacher generation, and gameplay out of this first model diagram. Raw logits are the
  clean boundary to those later parts of the story.

## First supervised run: the next teaching step

- Freeze the teacher corpus before showing optimisation: WDIT v3 generator `v0.1.0` supplies 52,726 training records
  (one is the opening state), 1,600 mini records, and 2,500 each for validation and untouched final test.
- Show the reader expanding a compact record through the same encoder used for future play. Its extra availability mask
  is separate from the candidate-bonus input and clears only guesses already made.
- The initial lesson can use one simple target: sparse cross-entropy from the teacher's top-ranked word. Adam starts at
  learning rate 0.001 with a deterministic nonzero seed; report loss and whether the teacher's first choice appears in
  the model's top 1, 5, or 16 suggestions.
- Explain checkpoints and scalar event files as repeatability tools, not evidence of a result. The code has automated
  plumbing checks, but the first actual run and all claims about its quality come later. Validation guides choices;
  final test stays sealed.

## Later structure

[To be expanded after the first experiment: baseline, learning curves, Go/CUDA boundary, failures and lessons, live
visualisation, and conclusion.]
