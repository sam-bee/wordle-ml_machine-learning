# Next steps

The policy model, canonical vocabulary and split contract, WDIT v3 teacher
corpus, shared state encoder, strict data reader, deterministic batching,
inspection command, and initial supervised-training proof are complete. The
[initial training proof report](../ml/initial-training-proof-report.md) records
the completed mini resume and fixed validation proof results.

## Approved fixed production continuation

One production run is approved without changing the model, objective, corpus,
or proof stages. It starts a new timestamped run from initialization through
`cmd/production -run-id=<timestamped-id>`, using the existing full training
split and opening-state sampling, FP32 Adam, batch size 256, constant learning
rate 3e-4, global clip norm 5, seed 20260808, and exactly 10,000 updates.
Validation and latest-checkpoint cadence are 100 updates, best is selected by
validation loss, and scalar telemetry cadence is 10 updates.

The production command is resumable with the same run ID and retains immutable
configuration and provenance, complete checkpoint/optimizer/sampler state,
TensorBoard events, training log, validation snapshots, and atomically written
final metrics. After successful training only, it independently reloads the
best checkpoint, reproduces validation metrics, evaluates all 100 validation
games, and writes `docs/ml/production-training-report.md` comparing that best
checkpoint with `proof-full-20260808`. Failures stop the chain with clear
status and logs; the run must not silently retry with changed settings.

After the generated production report is reviewed, keep any further experiment
bounded and validation-only until a new plan is approved.

The 100-solution final-test split remains sealed until model and training
decisions are finished. The completed proof does not establish broader
generalization. Gameplay, reinforcement learning, and a custom CUDA model are
not part of the next decision.
