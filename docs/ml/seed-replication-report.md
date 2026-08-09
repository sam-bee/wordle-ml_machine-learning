# Independent-seed production replication

<!-- seedreplicationreport: complete -->

This is one predeclared independent-initialization robustness check, not model tuning and not a comprehensive statistical study. The two runs use identical model, data, objective, optimiser, sampling, cadence, and 10,000-update configuration; only the recorded seed and run identity differ.

| Run | Seed | Final update / best update | Final validation loss | Best validation loss / top-1 / top-5 / top-16 | Games solved / mean guesses | Guess-count distribution (1..6) | Failed solutions |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| original `production-20260809-005026Z` | 20260808 | 10000 / 2200 | 4.6103 | 3.1341 / 0.510 / 0.611 / 0.664 | 97/100 / 3.680 | [0, 4, 42, 40, 10, 4] | HIPPY, MAMMY, UTTER |
| replication `seed-replication-20260809-132505Z` | 20260809 | 10000 / 2600 | 4.6436 | 3.1842 / 0.519 / 0.613 / 0.664 | 98/100 / 3.660 | [0, 5, 40, 42, 10, 3] | HIPPY, MAMMY |
| replication − original | — | — | +0.0332 | +0.0501 / +0.009 / +0.002 / +0.000 | +1 / -0.020 | [+0, +1, -2, +2, +0, -1] | — |

Under the predeclared lowest-best-validation-loss rule, `production-20260809-005026Z` has the lower validation loss. This statement does not change selected-model documentation.

## Failure sets

- Failed under both: HIPPY, MAMMY
- Original only: UTTER
- Replication only: none

## Paired validation states

| Both selected teacher top-1 | Original only | Replication only | Neither | Total |
| ---: | ---: | ---: | ---: | ---: |
| 1192 | 83 | 106 | 1119 | 2500 |

## Paired validation games

Guess delta is replication minus original. Failed games still record all six accepted guesses.

| Solution | Original | Replication | Original guesses | Replication guesses | Guess delta |
| --- | --- | --- | ---: | ---: | ---: |
| ADEPT | solved | solved | 3 | 3 | +0 |
| ADOBE | solved | solved | 4 | 4 | +0 |
| ALARM | solved | solved | 3 | 3 | +0 |
| ALOUD | solved | solved | 3 | 3 | +0 |
| AMISS | solved | solved | 2 | 2 | +0 |
| AVAIL | solved | solved | 3 | 3 | +0 |
| BADGE | solved | solved | 3 | 3 | +0 |
| BAGEL | solved | solved | 3 | 5 | +2 |
| BAKER | solved | solved | 4 | 5 | +1 |
| BLOWN | solved | solved | 3 | 3 | +0 |
| BOARD | solved | solved | 3 | 4 | +1 |
| BRAIN | solved | solved | 4 | 3 | -1 |
| BRIAR | solved | solved | 3 | 2 | -1 |
| BROAD | solved | solved | 3 | 4 | +1 |
| BURLY | solved | solved | 4 | 5 | +1 |
| CHAFF | solved | solved | 5 | 4 | -1 |
| CHASM | solved | solved | 3 | 3 | +0 |
| CHORD | solved | solved | 3 | 3 | +0 |
| CLACK | solved | solved | 4 | 4 | +0 |
| CLINK | solved | solved | 4 | 5 | +1 |
| CONCH | solved | solved | 3 | 3 | +0 |
| COUNT | solved | solved | 3 | 3 | +0 |
| CYCLE | solved | solved | 3 | 3 | +0 |
| DAUNT | solved | solved | 3 | 3 | +0 |
| DEALT | solved | solved | 3 | 3 | +0 |
| DEIGN | solved | solved | 3 | 3 | +0 |
| DRAWN | solved | solved | 4 | 4 | +0 |
| DREAM | solved | solved | 3 | 3 | +0 |
| DRUNK | solved | solved | 3 | 3 | +0 |
| EJECT | solved | solved | 4 | 4 | +0 |
| FAUNA | solved | solved | 4 | 4 | +0 |
| FILER | solved | solved | 5 | 3 | -2 |
| FOAMY | solved | solved | 4 | 4 | +0 |
| FORUM | solved | solved | 5 | 5 | +0 |
| FUNKY | solved | solved | 4 | 4 | +0 |
| GHOST | solved | solved | 4 | 3 | -1 |
| GNASH | solved | solved | 3 | 3 | +0 |
| GOING | solved | solved | 4 | 4 | +0 |
| GORGE | solved | solved | 4 | 4 | +0 |
| GRIEF | solved | solved | 4 | 4 | +0 |
| GRIMY | solved | solved | 4 | 4 | +0 |
| GROOM | solved | solved | 4 | 4 | +0 |
| GROPE | solved | solved | 3 | 3 | +0 |
| GUISE | solved | solved | 3 | 3 | +0 |
| HAIRY | solved | solved | 5 | 5 | +0 |
| HAVEN | solved | solved | 5 | 4 | -1 |
| HELIX | solved | solved | 4 | 4 | +0 |
| HELLO | solved | solved | 4 | 5 | +1 |
| HIPPY | failed | failed | 6 | 6 | +0 |
| HOWDY | solved | solved | 4 | 4 | +0 |
| JUICE | solved | solved | 4 | 4 | +0 |
| LASSO | solved | solved | 3 | 3 | +0 |
| LEAPT | solved | solved | 5 | 5 | +0 |
| LEPER | solved | solved | 4 | 4 | +0 |
| MAGIC | solved | solved | 3 | 3 | +0 |
| MAMMY | failed | failed | 6 | 6 | +0 |
| MAUVE | solved | solved | 4 | 4 | +0 |
| MERCY | solved | solved | 4 | 4 | +0 |
| METAL | solved | solved | 2 | 2 | +0 |
| MUSKY | solved | solved | 5 | 5 | +0 |
| NANNY | solved | solved | 6 | 6 | +0 |
| OCTET | solved | solved | 3 | 3 | +0 |
| ORDER | solved | solved | 4 | 4 | +0 |
| PALER | solved | solved | 4 | 4 | +0 |
| PAPER | solved | solved | 5 | 4 | -1 |
| PULPY | solved | solved | 4 | 4 | +0 |
| PUNCH | solved | solved | 4 | 4 | +0 |
| QUACK | solved | solved | 4 | 4 | +0 |
| RATIO | solved | solved | 4 | 4 | +0 |
| RATTY | solved | solved | 4 | 4 | +0 |
| RISKY | solved | solved | 3 | 3 | +0 |
| SATYR | solved | solved | 3 | 3 | +0 |
| SCION | solved | solved | 3 | 3 | +0 |
| SCOLD | solved | solved | 3 | 3 | +0 |
| SEIZE | solved | solved | 4 | 4 | +0 |
| SHARD | solved | solved | 3 | 3 | +0 |
| SHEAR | solved | solved | 3 | 4 | +1 |
| SHIRE | solved | solved | 2 | 2 | +0 |
| SHORE | solved | solved | 3 | 3 | +0 |
| SHORN | solved | solved | 4 | 4 | +0 |
| SMITH | solved | solved | 4 | 4 | +0 |
| SNAIL | solved | solved | 3 | 3 | +0 |
| SNEER | solved | solved | 4 | 4 | +0 |
| SPACE | solved | solved | 3 | 3 | +0 |
| SPELT | solved | solved | 2 | 2 | +0 |
| SPICE | solved | solved | 3 | 3 | +0 |
| SPIEL | solved | solved | 3 | 3 | +0 |
| STAID | solved | solved | 3 | 3 | +0 |
| STEAM | solved | solved | 5 | 4 | -1 |
| SUITE | solved | solved | 3 | 3 | +0 |
| THORN | solved | solved | 3 | 3 | +0 |
| TRAIL | solved | solved | 5 | 4 | -1 |
| TRICE | solved | solved | 3 | 3 | +0 |
| TRUCE | solved | solved | 4 | 4 | +0 |
| TWANG | solved | solved | 4 | 4 | +0 |
| UNMET | solved | solved | 4 | 5 | +1 |
| UTTER | failed | solved | 6 | 4 | -2 |
| VINYL | solved | solved | 3 | 3 | +0 |
| VODKA | solved | solved | 4 | 4 | +0 |
| VYING | solved | solved | 4 | 4 | +0 |

## Verified artifacts

- Original `/workspace/runs/production-20260809-005026Z` (seed 20260808; TensorBoard: `tensorboard --logdir /workspace/runs/production-20260809-005026Z/events`)
  - best checkpoint: `/workspace/runs/production-20260809-005026Z/checkpoints/best`; independent evaluation and trajectories: `/workspace/runs/production-20260809-005026Z/evaluations/best-games100.json` and `/workspace/runs/production-20260809-005026Z/evaluations/best-games100.jsonl`
  - commits: machine-learning `e3ed6ad7e0c58547e1932beb632c8ba750f1523b`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`
- Replication `/workspace/runs/seed-replication-20260809-132505Z` (seed 20260809; TensorBoard: `tensorboard --logdir /workspace/runs/seed-replication-20260809-132505Z/events`)
  - best checkpoint: `/workspace/runs/seed-replication-20260809-132505Z/checkpoints/best`; independent evaluation and trajectories: `/workspace/runs/seed-replication-20260809-132505Z/evaluations/best-games100.json` and `/workspace/runs/seed-replication-20260809-132505Z/evaluations/best-games100.jsonl`
  - commits: machine-learning `2718164bb80460757592b90aa86b96eb6d596018`; synthetic-data `456f60023fea5680a8f55db2c8e872cf802d6d20`; game-engine `57e6eaca35c7997364b1790c2a6cd33e0478ae0d`
- Paired teacher-top-1 artifact: `/workspace/runs/seed-replication-20260809-132505Z/evaluations/best-paired-top1.json`
- Validation split hash: `3cadd757ce9e6c57676358a8a13de4ef3d12fc2af7ba3033278c9926b867c019`; independent best-metric reproduction tolerance: `0.0002`

The sealed final-test split remained unopened: neither training, checkpoint selection, paired state comparison, gameplay evaluation, nor this report loads or evaluates final-test examples. All measurements above are from the fixed validation split.
