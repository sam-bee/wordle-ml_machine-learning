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

The objective is masked sparse categorical cross-entropy against the teacher's
top-ranked action. It uses FP32 Adam, global-gradient-norm clipping at 5, and
the fixed deterministic seed `20260808`. Metrics include loss and teacher
top-1, top-5, and top-16 agreement. The policy itself still returns raw logits
and has exactly 1,046,596 FP32 trainable parameters.

`cmd/train` selects one fixed proof stage with `-run-id` and `-stage`:

- `overfit`: 400 updates, batch size 128, learning rate 0.001;
- `mini`: 1,000 updates, batch size 128, learning rate 0.0003. A fresh mini
  run **must** stop normally with `-stop-at=500`, then resume the same ID
  without `-stop-at`;
- `full`: 2,000 updates, batch size 256, learning rate 0.0003.

Validation and checkpoint cadence are both 100 updates, while training scalar
telemetry is emitted every 10 updates. The standard-library `tensorboard`
package writes scalar and histogram event files, including training and
validation metrics, optimizer diagnostics, shortlist statistics, opening-state
diagnostics, parameter/logit/beta distributions, and gradient-norm
distributions.

Runs are self-contained below `runs/<run-id>/`: immutable `config.json` and
`metadata.json`, `run-state.json`, `training.log`, `final-metrics.json`,
`events/`, and `checkpoints/{initial,latest,best}`. The initial checkpoint is
saved at update zero; latest supports resume, while all three named snapshots
support independent reload.

The runner owns the overfit run-zero baseline: it independently reloads the
initial checkpoint and records its first 10 validation games in the overfit run
artifacts. `cmd/evaluate` therefore rejects an attempt to rerun
`overfit initial games10`. After mini passes, evaluate `mini latest games10`.
For a new proof, use fresh matching run IDs and this operator sequence:

```console
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=overfit-001 -stage=overfit
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=mini-001 -stage=mini -stop-at=500
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=mini-001 -stage=mini
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=mini-001 -checkpoint=latest -mode=games10
docker compose run --rm --no-deps wordleml go run ./cmd/train -run-id=full-001 -stage=full
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=initial -mode=games100
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=best -mode=games100
docker compose run --rm --no-deps wordleml go run ./cmd/evaluate -run-id=full-001 -checkpoint=best -mode=ablations
docker compose run --rm --no-deps wordleml go run ./cmd/report -overfit-run-id=overfit-001 -mini-run-id=mini-001 -full-run-id=full-001 -output=../docs/ml/initial-training-proof-report.md
```

Results are atomically recorded under `runs/<run-id>/evaluations/`; game
trajectories are JSONL and game summaries are added to TensorBoard. Finally,
`cmd/report` consumes the overfit, mini, and full run IDs, validates all proof
evidence and its four rendered rows (run-zero baseline, overfit, mini, full),
re-verifies TensorBoard event files for every proof stage and their
game-summary tags, re-derives the mini run's continuous event proof, then
atomically writes `docs/ml/initial-training-proof-report.md`. A failed gate
still produces a clearly marked incomplete report naming the stopped stage and
reason while the command returns an error; it does not overwrite an existing
successful report.

## Completed initial proof

The generated [initial training proof report](initial-training-proof-report.md)
records a passed mini stop/resume gate and a passed full proof. The best full
checkpoint reduced validation loss from 8.3005 to 3.1633 and raised validation
top-1 from 0.0056 to 0.5008. On the same fixed 100-game validation population,
it solved 97/100 games versus 4/100 for initialization, with mean guesses 3.65
versus 5.86. This is evidence for the bounded proof workflow only, not a claim
of generalization; the final-test split remains sealed.

The exact train/validation state-overlap audit is retained in every run: 190
of 2,445 unique validation encoded states also occur in training, and their
teacher top-1 labels agree. This is state-distribution overlap, not
solution-split leakage—the frozen solution IDs remain disjoint. No command
exposes the final-test split.
