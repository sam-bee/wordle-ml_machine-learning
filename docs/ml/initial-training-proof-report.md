# Initial training proof report

<!-- proofreport: complete -->

All required proof artifacts validated successfully. This report is generated from the immutable run artifacts; it does not rerun training or evaluation.

| Stage | Run | Updates | Train loss / top-1 / top-5 / top-16 | Validation loss / top-1 / top-5 / top-16 | Games win rate / mean guesses | Checkpoint | Pass |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| untrained baseline | `proof-overfit-20260808` | 0 | — | 8.3005 / 0.006 / 0.013 / 0.022 | 0.000 / 6.000 (suppressed raw I/R 0/2) | `initial` | passed |
| one-batch overfit | `proof-overfit-20260808` | 400 | 0.0954 / 0.989 / 0.989 / 0.990 | 17.2894 / 0.063 / 0.116 / 0.147 | — | `latest` | passed |
| mini | `proof-mini-20260808` | 1000 | 0.6592 / 0.866 / 0.884 / 0.898 | 13.1943 / 0.120 / 0.260 / 0.302 | 0.400 / 5.200 (suppressed raw I/R 0/9) | `latest` | passed |
| full | `proof-full-20260808` | 2000 | 2.4800 / 0.570 / 0.638 / 0.686 | 3.1633 / 0.501 / 0.605 / 0.660 | 0.970 / 3.650 (suppressed raw I/R 0/5) | `best` | passed |

## Artifacts

- untrained baseline: `/workspace/runs/proof-overfit-20260808` (TensorBoard: `tensorboard --logdir /workspace/runs/proof-overfit-20260808/events`)
  - checkpoint: `/workspace/runs/proof-overfit-20260808/checkpoints/initial`
  - commits: machine-learning `5f48086e1e26ccd28fafa9a9f148f61fba1d52fb`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`
- one-batch overfit: `/workspace/runs/proof-overfit-20260808` (TensorBoard: `tensorboard --logdir /workspace/runs/proof-overfit-20260808/events`)
  - checkpoint: `/workspace/runs/proof-overfit-20260808/checkpoints/latest`
  - commits: machine-learning `5f48086e1e26ccd28fafa9a9f148f61fba1d52fb`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`
- mini: `/workspace/runs/proof-mini-20260808` (TensorBoard: `tensorboard --logdir /workspace/runs/proof-mini-20260808/events`)
  - checkpoint: `/workspace/runs/proof-mini-20260808/checkpoints/latest`
  - commits: machine-learning `5f48086e1e26ccd28fafa9a9f148f61fba1d52fb`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`
- full: `/workspace/runs/proof-full-20260808` (TensorBoard: `tensorboard --logdir /workspace/runs/proof-full-20260808/events`)
  - checkpoint: `/workspace/runs/proof-full-20260808/checkpoints/best`
  - commits: machine-learning `5f48086e1e26ccd28fafa9a9f148f61fba1d52fb`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`

The full run includes independently reproduced best-checkpoint metrics, initial and best 100-game validation trajectories, and best-checkpoint ablations. The candidate-state ablation materially degrades validation performance; the best model improves that same 100-game population over the immutable initial checkpoint.

## Best-checkpoint ablations

| Ablation | Normal loss / top-1 / top-5 / top-16 | Ablated loss / top-1 / top-5 / top-16 | Effect (loss / top-1 / top-5 / top-16) |
| --- | --- | --- | --- |
| candidate state | 3.1633 / 0.501 / 0.605 / 0.660 | 9.7563 / 0.003 / 0.010 / 0.020 | +6.5930 / -0.498 / -0.595 / -0.639 |
| fixed turn | 3.1633 / 0.501 / 0.605 / 0.660 | 6.1600 / 0.442 / 0.514 / 0.542 | +2.9967 / -0.059 / -0.092 / -0.118 |
| no candidate bonus | 3.1633 / 0.501 / 0.605 / 0.660 | 9.0775 / 0.042 / 0.270 / 0.325 | +5.9141 / -0.459 / -0.336 / -0.335 |

## Deviations and warnings

- 190 of 2445 unique validation model states also occur in training; their teacher top-1 labels agree. This is state-distribution overlap, not solution-ID split overlap.
