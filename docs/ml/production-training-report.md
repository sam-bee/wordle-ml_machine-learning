# Production training report

<!-- productionreport: complete -->

This report validates the completed production run and independently reloaded best-checkpoint evaluation artifacts. It compares them with the retained 2,000-update initial proof; it does not rerun training or inference.

| Checkpoint | Run | Training updates / best update | Validation loss / top-1 / top-5 / top-16 | Games solved / mean guesses / failures | Guess-count distribution (1..6) |
| --- | --- | ---: | --- | --- | --- |
| production best | `production-20260809-005026Z` | 10000 / 2200 | 3.1341 / 0.510 / 0.611 / 0.664 | 97/100 (0.970) / 3.680 / 3 | [0, 4, 42, 40, 10, 4] |
| initial proof best | `proof-full-20260808` | 2000 / 2000 | 3.1633 / 0.501 / 0.605 / 0.660 | 97/100 (0.970) / 3.650 / 3 | [0, 4, 43, 41, 8, 4] |
| production − proof | — | — | -0.0292 / +0.009 / +0.006 / +0.004 | +0 / +0.030 / +0 | [+0, +0, -1, -1, +2, +0] |

## Verified artifacts

- Production: `/workspace/runs/production-20260809-005026Z` (TensorBoard: `tensorboard --logdir /workspace/runs/production-20260809-005026Z/events`)
  - best checkpoint: `/workspace/runs/production-20260809-005026Z/checkpoints/best`
  - independent evaluation: `/workspace/runs/production-20260809-005026Z/evaluations/best-games100.json`
  - commits: machine-learning `e3ed6ad7e0c58547e1932beb632c8ba750f1523b`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`
  - validation split hash: `3cadd757ce9e6c57676358a8a13de4ef3d12fc2af7ba3033278c9926b867c019`; best-metric reproduction tolerance: `0.0002`
- Initial proof reference: `/workspace/runs/proof-full-20260808`
  - best checkpoint: `/workspace/runs/proof-full-20260808/checkpoints/best`
  - independent evaluation: `/workspace/runs/proof-full-20260808/evaluations/best-games100.json`
  - commits: machine-learning `5f48086e1e26ccd28fafa9a9f148f61fba1d52fb`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`
  - validation split hash: `3cadd757ce9e6c57676358a8a13de4ef3d12fc2af7ba3033278c9926b867c019`; best-metric reproduction tolerance: `0.0002`

The sealed final-test split is not opened, evaluated, or represented by this report.
