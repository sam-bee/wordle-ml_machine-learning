`docs/` is the main folder for docs for the project. The contents should be an overview, not a detailed list of every
function. It is a high-level summary of what the project contains and what is being worked on. It should be a suitable
introduction for a person or agent, not a replacement for looking through the code for tiny details.

Current project documentation:

- [`development.md`](development.md) describes the Docker development stack and its smoke tests.
- [`data/overview-of-wordlists.md`](data/overview-of-wordlists.md) documents the model vocabularies and their provenance.
- [`ml/model-structure.md`](ml/model-structure.md) describes the GoMLX policy architecture and parameter count.
- [`ml/board-state-encoding.md`](ml/board-state-encoding.md) describes the four precomputed tensors consumed by the model.
- [`ml/inference-serving.md`](ml/inference-serving.md) describes the warm CUDA inference API and browser gameplay flow.
- [`ml/supervised-training.md`](ml/supervised-training.md) describes the frozen teacher corpus, loss, metrics, and
  completed proof/production evidence plus the fixed independent-seed replication workflow.
- [`ml/ppo-implementation.md`](ml/ppo-implementation.md) describes the optional, bounded PPO route, its isolated
  critic/checkpoint design, protected development split, and promotion protocol.
- [`ml/ppo-pilot-report.md`](ml/ppo-pilot-report.md) records both bounded PPO seeds, checkpoint acceptance evidence,
  and the final inconclusive comparison with the supervised actor.
- [`ml/initial-training-proof-report.md`](ml/initial-training-proof-report.md) is the generated evidence from the
  completed fixed validation proof.
- [`ml/production-training-report.md`](ml/production-training-report.md) is the generated validation-only comparison
  from the completed first fixed production run.
- [`ml/seed-replication-report.md`](ml/seed-replication-report.md) is the generated paired validation comparison from
  the completed fixed seed-20260809 replication.
- [`project-plan/next-steps.md`](project-plan/next-steps.md) records the completed fixed production continuation and
  subsequent review boundary;
  [`project-plan/codex-initial-training-proof-plan.md`](project-plan/codex-initial-training-proof-plan.md) records
  the implementation requirements and proof gates used for that run.
