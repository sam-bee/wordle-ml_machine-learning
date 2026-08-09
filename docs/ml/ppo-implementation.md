# Bounded PPO experimental route

This document describes the optional PPO reinforcement-learning route on the
`experiment/ppo-rl` branch. It is a deliberately small, rejectable experiment,
not a replacement for the completed supervised system. The supervised actor,
its training/evaluation commands, its deployment format, and the sealed final
test remain unchanged.

The pilot report is intentionally separate from this document. It records the
actual warm-up and candidate outcomes after the bounded run has completed;
this implementation document does not imply that PPO improved the actor.

## Baseline and compatibility

The fixed starting point is the selected supervised production actor:

```text
runs/production-20260809-005026Z/checkpoints/best
```

The PPO runner verifies the source run, its selected update (2,200), source
commit, and checkpoint hashes before it starts. The source is opened read-only
and is never a PPO checkpoint destination.

PPO keeps the proven `wordle_policy` actor separate from a new `ppo_critic`
value estimator. Loading the supervised checkpoint imports only the actor
variables, so the actor's raw logits match direct supervised inference before
any PPO optimiser update. The critic is separately checkpointed rather than
being added to the supervised graph: this avoids changing the known-good actor
structure, preserves existing checkpoint compatibility, and lets existing
supervised training continue without PPO machinery.

The critic consumes the established encoded state rather than a hidden answer
or a second board representation. It uses a small candidate-statistics
projection, a turn embedding, compact candidate/action-mask summaries, one
hidden layer, and a scalar output. The actor-only export contains only the
actor portion of a PPO checkpoint and reproduces the actor logits, so deployed
inference has no critic dependency.

## Split and environment boundary

[`data/rl/ppo-split-v1.json`](../../data/rl/ppo-split-v1.json) is a committed,
deterministic manifest derived only from the 2,109 existing supervised-training
solutions. It ranks words by SHA-256 of the domain-separated string
`wordleml-ppo-rl-split-v1\n` followed by the uppercase word, then reserves the
first 200 for PPO development evaluation and leaves 1,909 for rollout and
updates. The manifest records normalized-list hashes and is regenerated and
verified at startup.

The existing 100-solution validation split is not used for PPO rollouts,
critic targets, tuning, candidate selection, or PPO development evaluation.
The 100-solution final-test split remains sealed: the PPO command loads the
vocabulary through the final-test-safe loader and has no final-test mode.

PPO uses the existing Go game engine as the only authority for hidden-answer
state, feedback, legal guesses, solved/terminal state, and six-turn
termination. The shared `modelstate` encoder produces exactly the same four
model inputs used by the supervised actor. The hidden solution is retained
only as environment-side episode metadata and is never supplied to the actor
or critic.

The categorical action mask is also separate from the actor's learned
candidate-action feature. It starts with every frozen action-vocabulary word
available and clears only accepted prior guesses. Thus repeated accepted
guesses have zero probability and cannot be sampled, while useful
non-candidate probe words remain available whenever the engine accepts them.

## On-policy data and reward

For each PPO iteration, the last accepted actor is frozen as the old policy.
Hidden answers are sampled reproducibly from the rollout pool; the actor then
samples masked categorical actions through complete engine games. A transition
retains the encoded inputs, exact availability mask, action, old action
log-probability, old value, reward, terminal flag, episode ID, turn, and
environment-only solution ID. It also retains the return and advantage derived
from that completed episode.

Trajectories are used for one iteration only. They are discarded after critic
warm-up or after the candidate update; there is no replay buffer.

The initial objective is deliberately unshaped:

| Engine outcome | Reward |
| --- | ---: |
| Non-terminal accepted guess | -0.05 |
| Solving guess | +1.00 |
| Unsolved sixth guess | -1.00 |

This rewards completed-game success and, among solves, shorter games. It does
not include teacher agreement, candidate-count reduction, entropy reduction,
or any other shaping signal.

## PPO and value losses

With the stored mask applied identically to old, candidate, and supervised
reference logits, define:

```text
r_t(theta) = exp(log pi_theta(a_t | s_t) - log pi_old(a_t | s_t))
```

The actor maximises the standard clipped surrogate:

```text
mean(min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t))
```

The loss-minimising implementation adds entropy encouragement and a permanent
supervised-reference constraint:

```text
L_actor = -surrogate - c_entropy H(pi_theta)
          + c_ref KL(pi_supervised || pi_theta)
```

The reference KL is an exact categorical KL under the same stored availability
mask. PPO clipping limits movement from the immediately previous accepted
actor; this additional term and its hard gate limit cumulative drift from the
successful supervised actor.

The critic regresses realized return targets with:

```text
L_value = c_value * mean((V(s_t) - G_t)^2)
```

Returns use `G_t = r_t + gamma G_(t+1)`. Generalized advantage estimation is:

```text
delta_t = r_t + gamma (1-done_t) V(s_(t+1)) - V(s_t)
A_t = delta_t + gamma lambda (1-done_t) A_(t+1)
```

Advantages are normalized over each fresh rollout batch. All rewards, values,
returns, advantages, losses, gradients, and parameters have finite-number
checks. PPO epochs stop early when whole-rollout old-policy KL exceeds its
target.

## Fixed initial pilot configuration

[`configs/rl/ppo-pilot-v1.json`](../../configs/rl/ppo-pilot-v1.json) is the
versioned configuration. Its conservative initial settings are:

| Setting | Value |
| --- | ---: |
| Gamma / GAE lambda | 1.0 / 0.95 |
| Clip range / PPO epochs / minibatch | 0.10 / 4 / 256 |
| Actor / critic warm-up learning rate | 1e-5 / 1e-4 |
| Value / entropy coefficient | 0.5 / 0.001 |
| Maximum gradient norm | 1.0 |
| Old-policy KL target | 0.01 |
| Supervised-reference KL coefficient / hard maximum | 0.05 / 0.02 |
| Policy entropy floor | 0.10 |
| Critic warm-up | 3,072 games, 8 epochs |
| Pilot rollout | 512 rollout solutions x 2 games = 1,024 games |
| Pilot iterations / bootstrap samples | 1 / 10,000 |

Before an actor update, the critic is trained only on frozen-supervised-actor
rollouts. The actor checksum must remain unchanged. Warm-up records value loss,
explained variance, bias, opening-state prediction, predictions by turn, and
solved-versus-failed episodes. The actor remains frozen if held-out explained
variance does not both clear the configured minimum and improve meaningfully
over the untrained critic.

## Checkpoints, telemetry, and promotion

The required fresh `--run-dir` is isolated from the supervised checkpoint. Its
PPO layout is:

```text
<run-dir>/
  experiment-result.json
  critic-warmup.json
  checkpoints/
    supervised-baseline/metadata.json
    ppo/<run-id>/
      config.json  metadata.json  events/  accepted.json  best/
      iter-000/ ...
      iter-001/
        actor-critic/{actor,critic}/
        actor-only/
        evaluation.json
        iteration-state.json
```

Each candidate starts from the previous accepted actor/critic state. Candidate
artifacts are written before promotion; failed candidates stay inspectable but
do not replace `accepted.json` or `best`. The original supervised checkpoint
is therefore immediately usable after any failure. Actor and critic checkpoints
include the actor, critic, optimizer, global-step, seed, and PPO iteration
state needed to reproduce an iteration-boundary resume; an actor-only
checkpoint is also exported for deployment. The first bounded command starts
a fresh run and does not offer mid-minibatch command-line resume; that narrow
runner limitation is recorded explicitly rather than implying unsupported
recovery behavior.

TensorBoard events live under the PPO run's `events/` directory and are found
by the existing `runs/` TensorBoard mount when the generated run directory is
under `runs/`. Required scalar tags include episode return, solve rate,
solved/failure-counted guess metrics, policy/value loss, entropy, explained
variance, old-policy and supervised-reference KL, clip fraction, advantage and
return moments, gradient norm, rollout game/step counts, illegal/repeated
actions, actor delta, and opening action/probability. Every candidate also has
machine-readable greedy development evaluation JSON.

Promotion is based on greedy games over all 200 PPO-development solutions, not
sampled rollout return. A candidate must not reduce solved count, must improve
failure-counted mean guesses, must have no accepted illegal/repeated action,
and must satisfy numerical, old-policy-KL, supervised-reference-KL, and
entropy gates. Reports pair it both with the prior accepted actor and the
original supervised actor; they include per-solution improvements/worsening,
newly solved/failed IDs, histogram, opening action, and a seeded paired
bootstrap 95% interval.

The point estimate is classified as `rejected`, `inconclusive`, `promising`,
or `convincingly_improved`; promotion does not by itself make a result
convincing. If the first pilot produces an apparently successful PPO actor, it
must be rerun from the original supervised checkpoint with the second fixed
seed before PPO is described as successful. No long or open-ended production
run follows from this pilot.

## Running the bounded route

All commands run in the CUDA/GoMLX development container. After the normal
preflight, a fresh generated run can be started with:

```console
make format
make test
make smoke
docker compose run --rm --no-deps wordleml go run ./cmd/rl-train \
  --algorithm=ppo \
  --config=../configs/rl/ppo-pilot-v1.json \
  --supervised-checkpoint=../runs/production-20260809-005026Z/checkpoints/best \
  --run-dir=../runs/ppo-pilot-v1-seed-20260810
```

`--run-dir` must be new and must not overlap the supervised checkpoint. The
generated logs, checkpoints, events, and detailed evaluation artifacts stay
ignored beneath `runs/`; the committed configuration, split manifest,
implementation document, and eventual concise experiment report provide the
reviewable record. `make tensorboard` can view the generated event directory.

## Limitations

This is a conservative first ablation, not evidence that reinforcement
learning generally improves Wordle play. It uses only one small pilot batch,
completed-game reward, one fixed configuration, a development population
derived from supervised-training solutions, and greedy deployment evaluation.
The sealed validation and final-test populations remain outside the experiment.
Any claim beyond the report's bounded comparison requires separate approved
experiments.
