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

Model and training decisions then closed. The selected CUDA/cgo model performed
one authorized, post-selection aggregate gameplay evaluation of the 100 final
solutions: 97/100 solved, with 3.75 mean guesses. The result is in the
[final-test CUDA evaluation report](../ml/final-test-cuda-evaluation-report.md).
No tuning followed, and the final 2,500-record WDIT corpus remains unopened.

The completed proof and production run remain their historical validation-only
evidence; the final aggregate is one bounded held-out outcome, not a reason to
resume model selection. Gameplay, reinforcement learning, and further model
changes remain outside this completed workflow.
