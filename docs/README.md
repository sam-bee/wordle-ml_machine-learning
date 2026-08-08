`docs/` is the main folder for docs for the project. The contents should be an overview, not a detailed list of every
function. It is a high-level summary of what the project contains and what is being worked on. It should be a suitable
introduction for a person or agent, not a replacement for looking through the code for tiny details.

Current project documentation:

- [`development.md`](development.md) describes the Docker development stack and its smoke tests.
- [`data/overview-of-wordlists.md`](data/overview-of-wordlists.md) documents the model vocabularies and their provenance.
- [`ml/model-structure.md`](ml/model-structure.md) describes the GoMLX policy architecture and parameter count.
- [`ml/board-state-encoding.md`](ml/board-state-encoding.md) describes the four precomputed tensors consumed by the model.
- [`ml/supervised-training.md`](ml/supervised-training.md) describes the frozen teacher corpus, loss, metrics, and
  ready-but-not-run training command.
- [`project-plan/next-steps.md`](project-plan/next-steps.md) records only the immediate work for the first training
  experiment.
