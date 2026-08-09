# Next steps

The policy model, canonical vocabulary and split contract, WDIT v3 teacher
corpus, shared state encoder, strict data reader, deterministic batching,
inspection command, and initial supervised-training proof are complete. The
[initial training proof report](../ml/initial-training-proof-report.md) records
the completed mini resume and fixed validation proof results.

## Completed fixed production continuation

The approved production run completed without changing the model, objective,
corpus, or proof stages. `production-20260809-005026Z` started from fresh
initialization through `cmd/production`, using the existing full training split
and opening-state sampling, FP32 Adam, batch size 256, constant learning rate
3e-4, global clip norm 5, seed 20260808, and exactly 10,000 updates. Validation
and latest-checkpoint cadence were 100 updates, best was selected by validation
loss, and scalar telemetry cadence was 10 updates.

The production command is resumable with the same run ID and retains immutable
configuration and provenance, complete checkpoint/optimizer/sampler state,
TensorBoard events, training log, validation snapshots, and atomically written
final metrics. After successful training, it independently reloaded the update
2,200 best checkpoint, reproduced validation metrics, evaluated all 100
validation games, and wrote the
[production training report](../ml/production-training-report.md) comparing
that checkpoint with `proof-full-20260808`. Production validation loss was
3.1341 versus 3.1633 for the proof best; both solved 97/100 validation games,
with mean guesses 3.68 versus 3.65.

## Approved one-seed robustness replication

One additional validation-only experiment is approved: a fresh 10,000-update
production-style run with seed 20260809. The original seed-20260808 production
run remains immutable. Every other training, model, data, sampling, telemetry,
checkpoint, safety, and evaluation choice stays fixed; this is an
independent-initialisation robustness check, not tuning. A successful chain
will compare the two production seeds in the separate
`docs/ml/seed-replication-report.md`, including paired validation-state and
per-game results, without changing selected-model documentation.

Following this one replication, keep any further experiment bounded and
validation-only until another plan is approved.

The 100-solution final-test split remains sealed until model and training
decisions are finished. Neither the completed proof, the production run, nor
two production seeds establish broader generalization. Gameplay,
reinforcement learning, and a custom CUDA model are not part of the next
decision.
