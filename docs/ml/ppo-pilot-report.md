# Bounded PPO pilot report

<!-- ppo-pilot-report: complete -->

## Conclusion

**Overall conclusion: `inconclusive`.** Both one-iteration pilots produced a
strict-gate-acceptable, `promising` actor on the fixed 200-solution PPO
development split, but neither paired 95% bootstrap interval excluded zero.
The point estimates therefore did not distinguish PPO from the successful
supervised actor convincingly. No production PPO run was started.

The supervised baseline remains immutable and immediately usable. The two
small actor-only exports are retained as inspectable experimental artifacts;
they are not a replacement deployment checkpoint.

## Provenance, commands, and sealed populations

| Item | Value |
| --- | --- |
| Branch | `experiment/ppo-rl` |
| Base commit | `f4218934b077480fda3d1f024c63dad18c897d3b` |
| First-pilot execution commit | `f924650732371eeeb91ca6c52d52731460ef5ffd` |
| Second-pilot execution commit | `d526abc5e859c476745a06a2ad70b8da8f899187` |
| Final branch commit | Reported by `git rev-parse experiment/ppo-rl` in the final task handoff; a commit cannot embed its own content-derived SHA |
| First pilot | `ppo-pilot-v1-seed-20260810`, seed `20260810` |
| Independent second seed | `ppo-pilot-v1-seed-20260811`, seed `20260811` |
| Baseline actor | `production-20260809-005026Z`, best update `2200` |
| Baseline source commit | `e3ed6ad7e0c58547e1932beb632c8ba750f1523b` |

The baseline checkpoint was opened read-only at
`/workspace/runs/production-20260809-005026Z/checkpoints/best`.
Its checkpoint JSON was
`checkpoint-n0000022-20260809-005237-step-00002200.json`
(`443e172321456d5ed14ea7e61652ce60b5e0ef4693f9115a5691f11d04203206`), and
its binary was
`checkpoint-n0000022-20260809-005237-step-00002200.bin`
(`1f6e59a1175ff28548fced0573b10c22f6b1cd5deb31888abb3209bfd2b42fda`).

These are the exact host-shell pilot invocations. The service working
directory is `/workspace/wordleml`; each generated result also records the
same paths normalized to their absolute `/workspace/...` forms.

```console
docker compose run --rm --no-deps wordleml go run ./cmd/rl-train \
  --algorithm=ppo \
  --config=../configs/rl/ppo-pilot-v1.json \
  --data-dir=../data \
  --supervised-checkpoint=../runs/production-20260809-005026Z/checkpoints/best \
  --run-dir=../runs/ppo-pilot-v1-seed-20260810

docker compose run --rm --no-deps wordleml go run ./cmd/rl-train \
  --algorithm=ppo \
  --config=../configs/rl/ppo-pilot-v1-seed2.json \
  --data-dir=../data \
  --supervised-checkpoint=../runs/production-20260809-005026Z/checkpoints/best \
  --run-dir=../runs/ppo-pilot-v1-seed-20260811
```

The deterministic manifest was [`data/rl/ppo-split-v1.json`](../../data/rl/ppo-split-v1.json).
It was derived solely from the 2,109 supervised-training solutions:

| Population | Count | SHA-256 |
| --- | ---: | --- |
| Source training | 2,109 | `70184dfa5c291a73c8576f8ea6bbe041482890b63b68e965e9c94069577f7b78` |
| PPO rollout/update | 1,909 | `b715f1342ce0525b6a80ed014b7d8cd4b170cc4abe642d6f90d30507199cde76` |
| PPO development evaluation | 200 | `0a8b325b5599d33a42f9badf95f4a2a02251ac6ec9a9d008feb24675faef59d8` |

The existing 100-solution validation population was untouched. The 100-solution
final-test population remained sealed and unopened; both run metadata files
record `final_test_sealed_and_unopened: true`.

## Implementation and fixed configuration

The known-good supervised `wordle_policy` remains the actor. PPO adds a
separately checkpointed critic, so a supervised checkpoint still loads into
the actor without any graph or logits change and actor-only export has no
critic dependency. The critic has no hidden-answer input:

```text
encoded candidate-mask fraction
CandidateStats[209] -> Dense(64) -> ReLU
turn -> Embedding(6, 8)
encoded RemainingActionMask fraction
concatenate -> Dense(64) -> ReLU -> scalar V(s)
```

For an on-policy transition, with the exact stored availability mask reapplied:

```text
rho_t(theta) = exp(log pi_theta(a_t | s_t) - log pi_old(a_t | s_t))
L_actor = -mean(min(rho_t A_t, clip(rho_t, 0.9, 1.1) A_t))
          - 0.001 H(pi_theta)
          + 0.05 KL(pi_supervised || pi_theta)
L_value = 0.5 mean((V(s_t) - G_t)^2)
G_t = r_t + gamma G_(t+1)
delta_t = r_t + gamma (1-done_t)V(s_(t+1)) - V(s_t)
A_t = delta_t + gamma lambda (1-done_t)A_(t+1)
```

Advantages are normalized within each fresh rollout. The reward is exactly
`-0.05` for a non-terminal guess, `+1.00` for a solve, and `-1.00` for an
unsolved sixth guess. Old-policy KL stops PPO epochs at `0.01`; a separate
supervised-reference KL hard limit (`0.02`) prevents cumulative actor drift.

| Setting | Value |
| --- | ---: |
| Gamma / GAE lambda | `1.0` / `0.95` |
| Clip range / PPO epochs / minibatch | `0.10` / `4` / `256` |
| Actor / critic warm-up learning rate | `1e-5` / `1e-4` |
| Value / entropy coefficient | `0.5` / `0.001` |
| Maximum gradient norm | `1.0` |
| Supervised-reference KL coefficient / hard maximum | `0.05` / `0.02` |
| Policy entropy floor | `0.10` |
| Critic warm-up | 3,072 games, 8 epochs |
| Pilot rollout | 512 rollout solutions × 2 games = 1,024 games |
| Pilot iterations / paired bootstrap samples | 1 / 10,000 |

## Tests and numerical checks

The standard Docker checks passed before the pilots, and the formatting and
test checks were repeated after writing this report:

```console
make format
make test
make build
make smoke
docker compose run --rm --no-deps wordleml go vet ./...
```

`make test` ran `go test -p 1 ./...` in `/workspace/wordleml` and
`go test ./...` in `/workspace/web`; all packages passed. `make build`
included `cmd/rl-train`. `make smoke` passed on the approved NVIDIA GeForce
RTX 5070 Ti at compute capability 12.0, compiled CUDA only for `sm_120`, and
passed GoMLX's `xla:cuda` smoke test. `go vet ./...` also passed.

The PPO-focused tests cover supervised actor-logit compatibility, critic
isolation and actor-only export, exact resume state, zero-learning-rate actor
invariance, deterministic on-policy collection, masked sampling and repeated
guess exclusion, hidden-answer exclusion from observations, Wordle returns,
hand-computed GAE, finite advantage normalization, ratio-one before updates,
advantage direction, clipping, reference KL, checkpoint promotion/rollback,
and NaN/Inf rejection. Each warm-up and update recorded finite gradients and
parameters; both candidate artifacts report `numerically_stable: true`.

## Critic warm-up

The actor checksum did not change during either warm-up. The holdout used every
fourth sampled episode; the rest trained only the critic. Both runs cleared the
configured explained-variance floor (`0.01`) and the required improvement
(`0.01`), so the one bounded actor update was permitted.

| Seed | Transitions | Train / eval samples | Initial loss / EV / bias / opening V | Final loss / EV / bias / opening V | EV improvement | Gradient norm | Passed |
| ---: | ---: | --- | --- | --- | ---: | ---: | --- |
| 20260810 | 12,236 | 9,167 / 3,069 | 1.0823 / -0.1074 / -0.8531 / -0.0537 | 0.3034 / 0.0528 / 0.0066 / 0.7323 | 0.1602 | 0.1085 | yes |
| 20260811 | 12,326 | 9,276 / 3,050 | 1.2428 / -0.1888 / -0.9671 / -0.1825 | 0.2565 / 0.0109 / -0.0245 / 0.7450 | 0.1997 | 0.3117 | yes |

The following final holdout measurements give the requested turn-level view as
`count: predicted return / realized return / bias`.

| Turn | Seed 20260810 | Seed 20260811 |
| ---: | --- | --- |
| 1 | 768: 0.7323 / 0.7408 / -0.0085 | 768: 0.7450 / 0.7655 / -0.0205 |
| 2 | 767: 0.7955 / 0.7905 / +0.0050 | 767: 0.7666 / 0.8152 / -0.0486 |
| 3 | 739: 0.8266 / 0.8325 / -0.0059 | 740: 0.7634 / 0.8584 / -0.0951 |
| 4 | 502: 0.8101 / 0.8035 / +0.0066 | 498: 0.8832 / 0.8397 / +0.0435 |
| 5 | 194: 0.5255 / 0.5415 / -0.0160 | 190: 0.6721 / 0.6297 / +0.0423 |
| 6 | 99: 0.4250 / 0.1515 / +0.2735 | 87: 0.4589 / 0.2414 / +0.2175 |

Turn-level bias moved as follows from untrained to epoch eight; this includes
the initial per-turn warm-up evidence without repeating the realized-return
column above.

| Turn | Seed 20260810 initial → final bias | Seed 20260811 initial → final bias |
| ---: | ---: | ---: |
| 1 | -0.7945 → -0.0085 | -0.9480 → -0.0205 |
| 2 | -0.7251 → +0.0050 | -1.2183 → -0.0486 |
| 3 | -1.1332 → -0.0059 | -0.8753 → -0.0951 |
| 4 | -0.8776 → +0.0066 | -0.8354 → +0.0435 |
| 5 | -0.7488 → -0.0160 | -0.9816 → +0.0423 |
| 6 | -0.2873 → +0.2735 | -0.4232 → +0.2175 |

The critic learned a useful mean value fit but retained a substantial failed
episode overprediction. This is visible in the solved/failed aggregate below;
it is a caveat, not grounds to override the passed numerical warm-up gate.

| Seed | Kind | Initial: count, prediction / return / bias | Final: count, prediction / return / bias |
| ---: | --- | --- | --- |
| 20260810 | solved | 2,817, -0.0974 / 0.9220 / -1.0195 | 2,817, 0.7716 / 0.9220 / -0.1505 |
| 20260810 | failed | 252, -0.1179 / -1.1250 / +1.0071 | 252, 0.6373 / -1.1250 / +1.7623 |
| 20260811 | solved | 2,852, -0.1786 / 0.9221 / -1.1008 | 2,852, 0.7713 / 0.9221 / -0.1509 |
| 20260811 | failed | 198, -0.1665 / -1.1250 / +0.9585 | 198, 0.6703 / -1.1250 / +1.7953 |

## Baseline greedy development evaluation

The immutable supervised actor was evaluated greedily on all 200 PPO
development solutions before each pilot. Its results were identical apart
from the critic diagnostic (the actor itself was unchanged): 198 solved
(`0.990`), 3.5859 guesses among solved games, 3.6200 failure-counted guesses,
two failures, and histogram `[0, 2, 92, 93, 8, 5]` for actual guesses 1–6.
The opening was `ARISE` (action 159, probability `0.9946108`); there were zero
accepted illegal or repeated guesses. Mean policy entropy was `1.42019`.

## PPO candidates and acceptance

Each row is a newly collected 1,024-game rollout from the frozen supervised
actor, followed by four PPO epochs. Before every first optimizer update, the
stored old-policy ratio was exactly 1.0, with KL and clip fraction 0.0.

| Seed | Rollout: return / solve / solved guesses / FC guesses / steps | Rollout entropy / advantage mean±sd / illegal+repeat | Update: actor / critic loss | Update entropy / old KL / ref KL / clip fraction | Gradients actor / critic | Epoch stop / finite |
| ---: | --- | --- | --- | --- | --- | --- |
| 20260810 | 0.7467 / 0.94824 / 3.8857 / 4.0469 / 4,091 | 1.7235 / -0.0000±1.0000 / 0 | -0.00872 / 0.14674 | 1.7153 / 0.000593 / 0.000528 / 0.02909 | 1.0491 / 0.1074 | 4 epochs, no / yes |
| 20260811 | 0.7568 / 0.95312 / 3.8904 / 4.0361 / 4,085 | 1.7258 / -0.0000±1.0000 / 0 | -0.00832 / 0.12914 | 1.7114 / 0.000539 / 0.000477 / 0.02448 | 1.0959 / 0.1572 | 4 epochs, no / yes |

The remaining scalar update diagnostics are recorded here to make each
candidate auditable without opening TensorBoard. The first pair is the
unnormalized advantage mean / standard deviation; the second is return mean /
mean per-step reward. `Δactor` is the parameter delta from previous accepted /
original supervised actor.

| Seed | Pre-update loss / entropy / mean ratio / max \|ratio−1\| | Raw advantage μ / σ; return / step reward | PPO policy loss / mean ratio / max \|ratio−1\| | Update critic: count / value loss / EV / bias / opening V | Dev critic: count / value loss / EV / bias / opening V | Δactor previous / supervised |
| ---: | --- | --- | --- | --- | --- | --- |
| 20260810 | 0.00000 / 1.72347 / 1.00000 / 0.00000 | 0.00389 / 0.49107; 0.76283 / 0.18691 | -0.01440 / 0.99783 / 0.26956 | 4,091 / 0.28769 / 0.05606 / -0.00579 / 0.73043 | 721 / 0.08398 / 0.06687 / -0.12814 / 0.73043 | 0.26408 / 0.26408 |
| 20260811 | 0.00000 / 1.72581 / 1.00000 / 0.00000 | 0.00921 / 0.46065; 0.77767 / 0.18971 | -0.01426 / 0.99948 / 0.22395 | 4,085 / 0.25290 / 0.09326 / -0.00055 / 0.75022 | 720 / 0.09266 / -0.13001 / -0.10379 / 0.75022 | 0.24894 / 0.24894 |

Greedy candidate evaluation is against both the previous accepted checkpoint
and the original supervised checkpoint. Since each pilot has one iteration,
those two baseline comparisons are identical. `FC` is failure-counted mean
guesses; a negative paired difference favours PPO.

| Seed | Candidate solved / solved guesses / FC guesses / histogram | Candidate opening / probability | Eval entropy / old KL / ref KL / clip / critic EV | Pairs: improved / worsened / unchanged | Newly solved / failed IDs | Paired difference and 95% CI | Gates / status |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 20260810 | 198 / 3.5808 / 3.6150 / `[0,2,93,92,8,5]` | ARISE / 0.9940455 | 1.42008 / 0.000383 / 0.000383 / 0.01803 / 0.06687 | 200: 1 / 1 / 198 | none / none | -0.0050, [-0.0300, +0.0150] | all gates pass; promoted, `promising` |
| 20260811 | 198 / 3.5758 / 3.6100 / `[0,2,92,94,8,4]` | ARISE / 0.9947095 | 1.40485 / 0.000329 / 0.000329 / 0.01111 / -0.13001 | 200: 2 / 2 / 196 | none / none | -0.0100, [-0.0450, +0.0150] | all gates pass; promoted, `promising` |

For both candidates, all 200 evaluation games were numerically stable and
there were zero accepted illegal actions and zero accepted repeated actions.
Both held the solved count at 198 and strictly reduced FC mean guesses, so
they satisfy the within-run acceptance rules. Their old-policy KL was below
`0.01`, supervised-reference KL below `0.02`, and entropy far above `0.10`.
The pairwise bootstrap upper endpoint is nevertheless positive in both runs,
which is why neither result is `convincingly_improved`.

The candidate development critic EV is included above for transparency. It is
not a deployment acceptance metric; the second run's negative `-0.13001`
reinforces that the small critic is only adequate for this conservative pilot,
not evidence of a robust failure predictor.

## Checkpoints and generated artifacts

Generated artifacts remain ignored under `runs/`; no checkpoint or TensorBoard
event file is committed. The accepted actor-only exports are:

```text
/workspace/runs/ppo-pilot-v1-seed-20260810/checkpoints/ppo/ppo-pilot-v1-seed-20260810/iter-001/actor-only
/workspace/runs/ppo-pilot-v1-seed-20260811/checkpoints/ppo/ppo-pilot-v1-seed-20260811/iter-001/actor-only
```

Their associated actor-critic checkpoints, optimizer/iteration state, and
greedy evaluation JSON sit beside them at `iter-001/{actor-critic,
iteration-state.json,evaluation.json}`. The first run has actor checksum
`1a058b267e73eba8617c4c03687744e7bba26851a053016f5e9419c854e5ed60` and
critic checksum `23bf380bd853fa11861cc5ab971f6ba35a9dbf362d440402389633bba834495a`;
the second has actor checksum
`fb9bca0d2002fef6db17a828c6699836f7a404e552c9914fd2578dc0ca0acd8f` and
critic checksum `05645bd24ddb22262fd754761e875c2e2f1983c089d7e2a76f68193c2f78f735`.
Each run's `accepted.json` points only to its accepted `iter-001` artifacts;
the supervised checkpoint was never overwritten.

The primary records are:

- `/workspace/runs/ppo-pilot-v1-seed-20260810/{experiment-result.json,critic-warmup.json}`
- `/workspace/runs/ppo-pilot-v1-seed-20260811/{experiment-result.json,critic-warmup.json}`

TensorBoard events are at each
`checkpoints/ppo/<run-id>/events/` directory.
The two run directories occupy 39 MiB and 38 MiB respectively. A TensorBoard
scalar-tag query found every required `ppo/*` tag in both runs, including
failure-counted guesses and illegal/repeated accepted-action count; the extra
`ppo/step_reward_mean` tag records the requested per-step reward.

## Interpretation and caveats

The required second seed was run independently from the original supervised
checkpoint, using only the changed run identity and seed in
`ppo-pilot-v1-seed2.json`. It replicated the direction of the tiny point
estimate, but it also replicated the uncertainty: both intervals cross zero.
This is useful negative evidence, not a signal to extend the run.

Each standalone `experiment-result.json` records
`second_seed_replication_run: false` because one command does not orchestrate
another invocation. At the aggregate experiment level the replication **was**
run: the second command above starts again from the immutable supervised
checkpoint with seed `20260811`. This per-run field is a reporting limitation,
not evidence that the second pilot was skipped.

This experiment has only one PPO iteration per seed, a 1,024-game stochastic
rollout, one reward definition, one fixed hyperparameter configuration, and a
development population that was intentionally drawn from supervised training
solutions. It therefore says neither that PPO cannot help this model nor that
the actor-only exports generalize. In particular, late-turn and failed-game
critic bias remains material, and the bounded point estimates are much smaller
than their paired uncertainty.

Accordingly, PPO **matched but did not convincingly beat** the supervised
actor on this permitted development comparison. The final result is
`inconclusive`; the experimental branch is preserved for inspection and easy
rollback, and no sealed validation/final-test evaluation or production run was
performed.
