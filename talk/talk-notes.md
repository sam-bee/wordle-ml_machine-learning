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
- Show that GPU selection is a real engineering concern: Compose passes exactly
  one UUID-selected, approved RTX 5070 Ti or RTX 5050 (including the 5050
  Laptop GPU name) and rejects every other visible card, including an RTX 3060.
- Use the small CUDA smoke kernel to make `sm_120`, compute capability 12.0, host/device memory, and kernel launch syntax
  concrete before introducing model code.
- Follow it with the GoMLX Euclidean-distance graph to introduce symbolic graphs, XLA compilation, and the CUDA backend
  before introducing the full Wordle policy graph.
- Use the small standard-library TensorBoard writer as a reassuring Go detail:
  it writes ordinary scalar and histogram event files, not a new monitoring
  system. Actual learning claims still require a completed, passing proof run.
- Include its protobuf compatibility bug as an engineering lesson: the writer
  and its hand-written reader initially agreed on the wrong histogram field,
  while the real TensorBoard consumer did not. The fix pins the canonical
  field in a regression and checks output with TensorBoard itself.

## Wordle as a model problem

- Contrast the 2,309 possible solutions with the proposed 4,739-action vocabulary. A legal answer and a useful guess
  are related but distinct concepts.
- Use the fixed action vocabulary to make the model output concrete: one score per possible action, with a stable word
  at each output index.
- Keep the larger set of all game-legal guesses outside the initial model. This simplifies the demo while retaining
  every possible solution as an action.
- Split by solution, not by generated game state: 2,109 answers for training, 100 for validation while tuning, and 100
  held back for one final test. This prevents states for the same answer leaking across datasets.
- Be precise about a known caveat: the solution IDs remain disjoint, but 190
  of 2,445 unique validation encoded states also occur in training with
  agreeing teacher labels. Record that as state-distribution overlap, not as
  solution-split leakage.
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

## Fixed supervised proof workflow

- Freeze the teacher corpus before showing optimisation: WDIT v3 generator `v0.1.0` supplies 52,726 training records
  (one is the opening state), 1,600 mini records, and 2,500 each for validation and untouched final test.
- Show the reader expanding a compact record through the same encoder used for future play. Its extra availability mask
  is separate from the candidate-bonus input and clears only guesses already made.
- The fixed `overfit`, `mini`, and `full` stages make the demonstration
  repeatable: their batch sizes, learning rates, update budgets, seed, and
  checkpoint/validation cadence are recorded in each run rather than chosen on
  stage.
- Explain initial/latest/best checkpoints, continuous TensorBoard events, and
  per-run provenance as the evidence behind the completed
  [initial proof report](../docs/ml/initial-training-proof-report.md). The mini
  run passed its required stop at 500 and resume; the runner owns the overfit
  run-zero game baseline, rather than asking evaluation to rerun it.
- After full passes, show the fair same-population comparison: initial 100
  validation games, best 100 validation games, then best-checkpoint ablations.
  The report command consumes the three proof run IDs, verifies four rendered
  rows, re-verifies every stage's TensorBoard event files and game-summary tags,
  and writes the checked-in Markdown report without rerunning training or
  evaluation.
- The completed fixed validation proof reduced loss from 8.3005 to 3.1633 and
  increased top-1 from 0.0056 to 0.5008. Its best checkpoint solved 97/100 of
  the fixed validation games versus 4/100 at initialization, reducing mean
  guesses from 5.86 to 3.65. Present this as a bounded proof result, not a
  generalization claim: validation guides choices and final test stays sealed.
- The proof was followed by one fresh, fixed production continuation: the same
  full training split, encoder, policy, objective, opening-state sampling,
  seed, optimizer, batch size, learning rate, and clipping; the only
  configuration difference was the 10,000-update budget. That was a longer
  confirmation run, not a hyperparameter search or a retroactive change to
  the proof.
- Explain the production handoff in engineering terms: immutable provenance;
  initial/latest/best checkpoints; complete optimizer and sampler resume
  state; telemetry every 10 updates; validation and latest snapshots every
  100; numerical-safety checks; and a separate post-training reload before
  all 100 validation games are played.
- The [production report](../docs/ml/production-training-report.md) selected
  update 2,200 from the 10,000-update run. Against the retained proof best,
  validation loss improved from 3.1633 to 3.1341 and top-1/top-5/top-16 moved
  from 0.5008/0.6052/0.6596 to 0.5100/0.6108/0.6640. Both solved 97/100
  validation games; mean guesses rose slightly from 3.65 to 3.68. That contrast
  is a useful lesson: a small validation metric improvement did not improve
  this gameplay success rate. Keep the claim validation-only and the final
  test sealed.
- For the live demo, use the web application's run picker to contrast the
  retained proof and production best checkpoints. Changing the selection
  restores and warms that checkpoint on CUDA before it becomes active; the
  original model remains available if loading fails. Then select a validation
  solution. The Go server proxies same-origin requests to the internal
  inference service; the Go host advances the authoritative game and encodes
  each state, while up to six policy forward passes execute through GoMLX on
  CUDA. The browser animates the returned trajectory without receiving GPU
  access.

## Later structure

[To be expanded after the first experiment: baseline, learning curves, Go/CUDA boundary, failures and lessons, and
conclusion.]
