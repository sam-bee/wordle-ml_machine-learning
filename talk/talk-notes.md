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
- Follow it with the GoMLX Euclidean-distance graph to introduce symbolic graphs, XLA compilation, and the reference
  CUDA backend before introducing the full Wordle policy graph. Make clear that the later hand-written CUDA/cgo demo is
  a separate inference route, not a claim that GoMLX has disappeared.
- Use the small standard-library TensorBoard writer as a reassuring Go detail:
  it writes ordinary scalar and histogram event files, not a new monitoring
  system. Actual learning claims still require a completed, passing proof run.

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
- Keep the existing port-8082 GoMLX visualiser as a useful contrast: its run
  picker selects compatible checkpoints and the browser proxies to an internal
  GoMLX/XLA service. The direct port-8083 CUDA/cgo demo deliberately differs:
  it serves one exported checkpoint from one Go process, has no HTTP inference
  proxy and no CUDA hot swap, and visibly names the `cuda-cgo` backend and GPU.
  In either demo, select only an advertised validation solution; the Go host
  advances the authoritative game and the browser animates the completed
  trajectory without receiving GPU access.

## Go as control plane, CUDA as numerical data plane

### Ownership and legality

- Say “control plane” in the ordinary engineering sense, not as ML jargon:
  Go owns the web application, model artifact validation, vocabulary hashes,
  Wordle feedback/rules, shared board-state encoding, and complete-game
  orchestration.
- CUDA receives four numeric tensors and returns 4,739 raw FP32 logits. It is
  deliberately not told which actions Go has suppressed because they were
  guessed already; it has no Wordle rule implementation and no CUDA-side
  argmax.
- Revisit the two masks. `remainingActionMask` is a learned candidate bonus
  input; it is not a hard-mode or legal-action mask. Go applies the separate
  availability mask afterwards, suppresses repeated guesses, and retains the
  existing lower-action-ID tie behavior.

### The cgo boundary

- Show the plain C ABI around CUDA C++: an opaque C-owned model handle is
  created once, accepts four fixed input arrays, and returns one logit array.
  It is deliberately small enough to fit on a slide.
- One complete forward pass crosses cgo once. Calling from Go once per layer
  would hide the actual boundary behind plumbing and make every layer a
  host/device coordination point.
- There is no callback from C to Go and C never retains a Go pointer. The Go
  wrapper uses `runtime.KeepAlive`; the native call is synchronous from Go’s
  perspective and returns only after the logits copy and stream synchronization.
- A dedicated goroutine calls `runtime.LockOSThread`, creates the handle,
  serializes inference, and destroys it on that same OS thread. This gives the
  CUDA context a simple owner and makes a `wordle-gpu` lane recognizable in
  Nsight Systems.

### Threads, warps, blocks, and grids

- Use the actual output-layer launch: **4,739 blocks**, **128 threads per
  block**, **4 warps per block**, and **one block computes one possible-word
  logit**.
- That is 606,592 logical threads in the launch, not 606,592 CPU-like threads
  simultaneously resident on the GPU. Hardware schedules blocks onto SMs in
  waves; occupancy, registers, and available resources affect how many are
  active at once.
- In each dense block, every thread accumulates a strided part of one dot
  product in a register. Warp shuffles reduce those partials; four warp
  leaders use a tiny shared-memory reduction before one thread writes the
  output. This connects a familiar Go loop to the GPU mapping without turning
  the talk into a general CUDA course.

### Memory

- Draw the actual data path: Go slices in host memory; three input copies to
  device global memory; one persistent global-memory weight allocation; device
  buffers for hidden state, residual scratch, beta, and output; then one logits
  copy back to host memory.
- Weights and persistent buffers allocate at model creation, not within each
  inference. Global-memory reads may use caches. Per-thread partial sums live
  in registers; the four warp subtotals use shared memory.
- Be precise that shared memory belongs to one block, not the whole GPU. Local
  memory is what spills from registers to device memory; do not describe it as
  fast local CPU stack storage. Use the Nsight Compute report to establish any
  actual register/shared/local values.

### Profiling

- Nsight Systems is the wide-angle timeline: host thread, NVTX inference range,
  HtoD copies, seven named kernels, DtoH copy, and synchronization.
- Nsight Compute is the microscope for one kernel, here
  `policy_logits_with_bonus`: launch configuration, occupancy, memory workload,
  register/shared use, and source correlation.
- The collected Systems report now shows `wordle-gpu`, its NVTX range, all
  three input copies, all seven kernels, the output copy, and final
  synchronization. Explain that a checked CUDA-device reset at teardown gives
  CUPTI a finalization point; it is not work performed during inference.
- The collected Compute report is a concrete but narrow observation: the
  4,739 × 128 policy launch took 11.36 µs, used 40 registers/thread, had 16 B
  static plus 1.02 KB driver shared memory per block, no local/shared spills,
  and achieved 69.89% occupancy against 100% theoretical. Use those values to
  explain a report, not to claim a universal optimum or speedup.
- Batch-one inference is intentionally a useful teaching case because cgo
  crossing, kernel launches, and transfers can matter. CUDA is not faster just
  because a GPU was used; use the benchmark and profiler reports before making
  a performance claim.

### Demo sequence

1. Start with the browser: the checked-in capture shows validation word `ADEPT`
   solved in three guesses while the page identifies hand-written CUDA via cgo,
   the exported model, and the actual GPU.
2. Show the real Go wrapper: one cgo crossing, not a fake slide-only snippet.
3. Show the real C ABI and CUDA launch, then explain grid, block, warp, thread,
   output-major weights, and the seven readable kernel names.
4. Open Nsight Systems and zoom into one `wordle_infer` range on `wordle-gpu`.
5. Open Nsight Compute at `policy_logits_with_bonus` and show launch,
   occupancy, memory, and source views.
6. Conclude with the measured parity and benchmark evidence, not a presumed
   speedup: 32/32 golden top-1/top-5/selected actions and 100/100 validation
   trajectories match; the 200-call warm benchmark mean was 94,945 ns. Nsight
   GUI screenshots remain a manual capture step even though report artifacts
   exist.

## Later structure

[Develop this into slides around the browser result, model diagram, Go/CUDA
boundary, verification evidence, profiler evidence, failures and lessons, and
conclusion. Keep the Wordle game visible at each transition.]
