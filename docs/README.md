`docs/` is the main folder for docs for the project. The contents should be an overview, not a detailed list of every
function. It is a high-level summary of what the project contains and what is being worked on. It should be a suitable
introduction for a person or agent, not a replacement for looking through the code for tiny details.

Current project documentation:

- [`development.md`](development.md) describes the Docker development stack and its smoke tests.
- [`data/overview-of-wordlists.md`](data/overview-of-wordlists.md) documents the model vocabularies and their provenance.
- [`ml/model-structure.md`](ml/model-structure.md) describes the GoMLX policy architecture and parameter count.
- [`ml/board-state-encoding.md`](ml/board-state-encoding.md) describes the four precomputed tensors consumed by the model.
- [`ml/inference-serving.md`](ml/inference-serving.md) describes the retained warm GoMLX/XLA inference API and browser
  gameplay flow.
- [`ml/cuda-cgo-inference.md`](ml/cuda-cgo-inference.md) describes the offline GoMLX exporter, the GoMLX-free
  hand-written CUDA/cgo runtime, its artifact contract, verification, profiling, and direct browser demo.
- [`ml/supervised-training.md`](ml/supervised-training.md) describes the frozen teacher corpus, loss, metrics, and
  completed initial supervised-training proof, with a link to its report.
- [`ml/initial-training-proof-report.md`](ml/initial-training-proof-report.md) is the generated evidence from the
  completed fixed validation proof.
- [`ml/production-training-report.md`](ml/production-training-report.md) is the generated validation-only comparison
  from the completed first fixed production run.
- [`project-plan/next-steps.md`](project-plan/next-steps.md) records the completed fixed production continuation and
  subsequent review boundary;
  [`project-plan/codex-initial-training-proof-plan.md`](project-plan/codex-initial-training-proof-plan.md) records
  the implementation requirements and proof gates used for that run.
- [`project-plan/cuda-cgo-inference-working-notes.md`](project-plan/cuda-cgo-inference-working-notes.md) records the
  pre-implementation package map and selected checkpoint provenance for the CUDA/cgo work.
