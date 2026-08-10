# One-time final-test CUDA gameplay evaluation

This document records the completed, one intentional post-selection gameplay
evaluation of the 100 final solutions through the hand-written CUDA/cgo
backend. Its source is the sanitized
[`final-test-evaluation.json`](../../artifacts/cuda-cgo/final-test-evaluation.json);
that file contains aggregate facts only and no words, guesses, trajectories, or
failed-solution list.

This was the first intentional post-selection gameplay scoring of the final
population, not the first historical read of its word list. Earlier proof and
production workflows could load that fixed list for vocabulary and split-hash
context, but they did not score it or use it for model decisions. The separate
final WDIT v3 test corpus of 2,500 records was not opened. No tuning followed
this result.

## Fixed identity

| Field | Value |
| --- | --- |
| Completion time (UTC) | `2026-08-09T23:54:12.480769529Z` |
| Evaluator commit | `b5703a74e63e37ac0cbf54aae4ca79baa9799152` |
| Model | `seed-replication-20260809-132505Z`, `best` update 2,600 |
| Training commit | `2718164bb80460757592b90aa86b96eb6d596018` |
| Artifact format / parameters | `wordle-cuda-f32-v1` / 1,046,596 FP32 |
| Weight SHA-256 | `b78dc980505998d9dd40551ef4d24788b8378be63e4d09fb90aa0a8be83c870d` |
| Final-solution-list SHA-256 | `978a25608a96370b3e26cc8621e9f2cc83ad2d581d07b4b23546b0b4ccdec130` |
| CUDA device | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 |
| CUDA runtime / driver | 13.1 / 13.2 |

## Evaluation method

The executed command was:

```console
make cuda-cgo-final-test MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

This is an evidence record, not an instruction to rerun the command. Before
reading final-solution membership, the target checked every runtime input
against evaluator commit `b5703a7`, rebuilt and audited the Go/cgo/CUDA binary,
validated the fixed model identity and approved GPU, and created the report
path with an exclusive one-shot claim. It then verified the expected final-list
hash and count, played exactly 100 games with Go retaining rules and legal
selection, and replaced the claim with this sanitized aggregate. The existing
claim makes a later invocation refuse before another final-list read.

The build's source revision and embedded evaluator revision both match the
commit above. The only deliberately excluded working-tree change was the
repository's non-runtime `AGENTS.md` operator-instruction file; no compiled or
runtime input differed from the recorded commit.

## Aggregate gameplay result

| Measure | Result |
| --- | ---: |
| Final solutions scored | 100 |
| Games solved | 97 |
| Solved fraction | 0.970 |
| Mean guesses | 3.75 |
| Failures | 3 |
| Guess-count distribution (1..6) | [0, 0, 45, 41, 8, 6] |
| Invalid selections | 0 |
| Suppressed raw-top selections | 6 |
| Repeated selections | 6 |

Failed games count as six guesses in the distribution and mean. These are
aggregate results for this fixed, held-out 100-solution gameplay population;
they do not revise the historical validation-only proof, production, or
CUDA-parity claims. The final WDIT corpus remains unopened.
