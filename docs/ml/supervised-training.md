# Initial supervised-training boundary

The first run uses a frozen, offline WDIT v3 corpus produced by
`wordle-ml_synthetic-data-creation` release `v0.1.0`. The checked-in artifacts
contain 52,726 train records, including one opening record; 1,600 mini
records; and 2,500 records each for validation and final test. The final-test
files are deliberately not opened by default inspection or training work.

`imitationdata` validates each binary file and its JSON sidecar against the
checked-in vocabulary and solution-split hashes before expanding a record.
The shared `modelstate` encoder then produces the policy's four inputs. Its
candidate-action mask remains a learned bonus input, not a legality rule.

Training supplies one additional, separate availability mask. It begins with
all 4,739 actions available and clears only previous history guesses. The
training wrapper applies that mask to raw policy logits before the loss, so
non-candidate probe words remain playable unless they have already been used.

The initial objective is sparse categorical cross-entropy against the
teacher's top-ranked action. It uses Adam with a default learning rate of
0.001 and a deterministic nonzero seed. The first metrics are loss and teacher
top-1, top-5, and top-16 agreement. The policy itself still returns raw logits
and has exactly 1,046,596 FP32 trainable parameters.

Checkpoints make a run resumable, while the `tensorboard` package writes scalar
TensorBoard event files using only the Go standard library. Generated
checkpoints and event files are ignored by Git. This plumbing has automated
tests; no training experiment, learning curve, or model-quality result has
been produced yet. The first experiment must use train and validation only;
final test remains sealed until evaluation.

The command defaults to a zero-step dry run, so inspecting its configuration
does not load data, initialize CUDA, or create output files:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/train
```

An explicit positive `-steps` value enables the bounded training path. It reads
only train and validation, logs under `data/tensorboard/first-run`, and resumes
or saves under `data/checkpoints/first-run`. Starting that first experiment is
deliberately the next task, not part of this implementation phase.
