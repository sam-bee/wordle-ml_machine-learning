# Codex Instructions: Prove the Initial Training System

## Goal

Implement and run the **early supervised-training proof phase** for the Wordle ML project.

This work begins **after the pre-training implementation plan is complete** and the model can already accept encoded examples and produce logits. The purpose here is not to train the final model. The purpose is to prove that:

- the selected model architecture can learn;
- the data pipeline, masking, loss and optimizer are correct;
- checkpointing and resume work;
- TensorBoard telemetry is useful;
- a trained checkpoint can be used to play complete Wordle games through the shared game engine.

Keep this phase deliberately small and easy to diagnose.

The relevant repositories are:

- `https://github.com/sam-bee/wordle-ml_machine-learning`
  - model, training loop, evaluation, checkpoints and TensorBoard;
- `https://github.com/sam-bee/wordle-ml_synthetic-data-creation`
  - teacher-labelled imitation-learning datasets;
- `https://github.com/sam-bee/wordle-ml_game-engine`
  - authoritative game state, candidate-shortlist behaviour and complete-game evaluation.

Do not redesign the selected model architecture unless a hard engineering failure proves that it is necessary.

Do not begin a long production training run as part of this task.

---

## 1. Implement one simple baseline training configuration

Start with a deliberately boring configuration:

```text
precision:              float32
objective:              masked cross-entropy against teacher top-1
optimizer:              Adam
learning rate:          3e-4
batch size:             256
gradient clipping:      global norm 5.0
weight decay:           none
learning-rate schedule: constant
training seed:          one fixed seed
validation cadence:     every 100 updates
checkpoint cadence:     every 100 updates
```

For the single-batch overfit test below, use:

```text
batch size:    128
learning rate: 1e-3
```

Do not introduce mixed precision, schedulers, label smoothing, soft targets, hyperparameter searches, multiple seeds or RL during this proof phase.

Use the GoMLX version already pinned by the project. Do not upgrade GoMLX while performing these experiments.

### Opening state

There is effectively one empty-board Wordle state, so ordinary uniform record sampling can underrepresent it.

During these proof runs, ensure that **one opening-state training example is included in each batch**. Track opening-state performance separately in telemetry.

---

## 2. Implement reproducible run directories

Each training run must produce a self-contained run directory, for example:

```text
runs/<run-id>/
    config.json
    metadata.json
    events/
    checkpoints/
        latest/
        best/
    final-metrics.json
    validation-games.jsonl
    training.log
```

Record enough metadata to reproduce the run.

At minimum include:

- Git commit hash for `wordle-ml_machine-learning`;
- Git commit hash for `wordle-ml_synthetic-data-creation`;
- Git commit hash for `wordle-ml_game-engine`;
- dataset format/version;
- dataset file hashes;
- vocabulary hash;
- training/validation/test split hashes;
- model parameter count;
- Go version;
- GoMLX version;
- execution backend;
- GPU details;
- CUDA/PJRT details where relevant;
- random seed;
- complete effective training configuration.

---

## 3. Implement full checkpoint and resume support

A checkpoint must contain enough state for genuine continuation rather than merely reloading model weights.

Persist:

- model parameters;
- Adam optimizer state/moments;
- global update number;
- random-number state where applicable;
- dataset epoch;
- shuffle seed/state;
- current position in the shuffled dataset;
- current best validation result.

Resuming must:

- continue from the correct global update;
- continue the same TensorBoard run without resetting the x-axis;
- restore optimizer state;
- restore dataset position sufficiently well to continue the run rather than silently restarting an epoch.

Later in this task we will explicitly stop and resume a training run to verify this.

---

## 4. Implement TensorBoard telemetry

Use TensorBoard as the primary training telemetry.

Log lightweight training scalars approximately every 10 updates and validation metrics every 100 updates.

### Training metrics

At minimum:

```text
train/loss
train/top1_accuracy
train/top5_accuracy
train/top16_accuracy
```

The top-k metrics should compare model predictions with the ranked teacher information available in the synthetic dataset where possible.

### Validation metrics

At minimum:

```text
validation/loss
validation/top1_accuracy
validation/top5_accuracy
validation/top16_accuracy
```

Also break validation results down by:

- turn number;
- candidate-shortlist size.

Use shortlist-size buckets approximately like:

```text
1
2-5
6-20
21-100
>100
```

Also record whether the model's highest-ranked guess occurs anywhere in the teacher's stored top 16.

### Optimizer/model metrics

At minimum:

```text
optimizer/learning_rate
optimizer/global_gradient_norm
optimizer/parameter_norm
optimizer/update_to_parameter_norm

model/output_entropy
```

If the selected architecture contains the learned candidate bonus / `beta`, also log useful summary statistics for it.

### Data metrics

At minimum:

```text
data/epoch
data/examples_consumed
data/shortlist_size_mean
data/shortlist_size_min
data/shortlist_size_max
```

### Performance metrics

At minimum:

```text
performance/examples_per_second
performance/batch_duration
performance/input_wait_duration
performance/validation_duration
```

### Opening-state metrics

At minimum:

```text
opening/loss
opening/teacher_rank
```

Also make the current highest-ranked opening guess easy to inspect.

### Game-evaluation metrics

When full-game evaluation is run, record:

```text
games/solved_fraction
games/mean_guesses
games/failures
```

Record the guess-count distribution as well.

### Histograms

Do not write expensive histograms every training update.

At validation/checkpoint intervals, useful histograms include:

- logits;
- learned `beta`, if present;
- parameter values;
- per-layer gradient norms.

---

## 5. Run zero: establish the untrained baseline

Before performing the first optimizer update, run a complete validation pass.

Record:

- initial validation loss;
- validation top-1 agreement;
- validation top-5 agreement;
- validation top-16 agreement;
- results by turn;
- results by shortlist-size bucket;
- opening-state prediction;
- parameter statistics;
- logit statistics.

Also play **10 fixed validation games** with the untrained model and save their trajectories.

These same fixed games should be available for later before/after comparison.

Add hard sanity checks that verify:

- every teacher target is a valid vocabulary ID;
- every teacher target is legal/unmasked for its example;
- masked logits cannot be selected;
- losses are finite;
- gradients are finite;
- validation cannot update model parameters;
- saving and immediately reloading the untrained checkpoint reproduces the same predictions to expected numerical tolerance.

For reference, a perfectly uniform distribution across 4,739 outputs has cross-entropy of roughly `ln(4739) ~= 8.46`. Do not treat that as an exact expected value; just use it as a useful sanity check.

Commit any implementation changes before proceeding to the learning tests.

---

## 6. Run one: prove the model can memorise one batch

Construct one deterministic batch containing **128 unique states**.

The batch should include examples across multiple:

- turns;
- shortlist sizes.

Deduplicate encoded states. If identical encoded states somehow have conflicting labels, stop and investigate instead of training on them.

Train repeatedly on only this batch:

```text
batch size:       128
updates:          up to 400
learning rate:    1e-3
```

### Pass conditions

This run passes if:

- training loss drops by approximately 90% or more;
- teacher top-1 agreement on the batch reaches at least 95%;
- gradients remain finite;
- parameters remain finite;
- actual model weights change, not merely an output bias;
- saving and reloading the trained checkpoint preserves predictions;
- TensorBoard contains sensible curves for the run.

If this test fails, **stop the training sequence**.

Do not compensate by training longer or generating more data.

Investigate the engineering path first:

- labels;
- vocabulary IDs;
- legality masks;
- masked cross-entropy;
- gradient construction;
- optimizer application;
- parameter registration;
- batch construction.

Only proceed once single-batch overfitting works.

---

## 7. Run two: prove learning on the mini dataset

Use the deterministic mini dataset from `wordle-ml_synthetic-data-creation`.

This run is still an engineering test and is allowed to overfit.

Suggested configuration:

```text
dataset:          mini
batch size:       128
updates:          1,000
learning rate:    3e-4
validation:       every 100 updates
checkpoint:       every 100 updates
```

### Explicit checkpoint/resume test

At update 500:

1. write a complete checkpoint;
2. terminate the training process normally;
3. restart training from that checkpoint;
4. continue to update 1,000.

Verify that:

- TensorBoard continues from step 500 rather than restarting at zero;
- Adam state is restored;
- the loss curve does not show behaviour consistent with a freshly reset optimizer;
- dataset shuffle/position resumes correctly;
- run metadata still identifies this as one logical training run.

If practical, record the next 20 example IDs after update 500 in both:

- an uninterrupted reference run;
- the stop-and-resume run.

They should match if the dataset state is expected to be exactly reproducible.

Do not require bit-for-bit equality of all GPU floating-point results if the backend does not guarantee it.

### Mini-run pass conditions

The run should show:

- training loss falling to less than roughly half its initial value;
- a large increase in training top-1 agreement;
- a large increase in training top-16 agreement;
- successful repeated validation passes;
- successful checkpoint save and resume;
- complete TensorBoard output;
- no NaNs or infinities.

Then load the resulting checkpoint independently and play the same **10 fixed validation games** used for the untrained baseline.

All games must execute without:

- invalid actions;
- masked actions;
- state-encoding failures;
- vocabulary-ID mismatches;
- game-engine integration failures.

Validation quality does not need to be impressive yet.

If this run fails, stop and diagnose it before attempting full-data training.

---

## 8. Run three: short full-data proof

Once the single-batch and mini-data gates pass, perform the first short run against the full training dataset.

Use the established split:

```text
training solutions:   2,109
validation solutions: 100
test solutions:       100
```

The test split must remain untouched during this phase.

Suggested configuration:

```text
dataset:             full training split
batch size:          256
updates:             2,000
learning rate:       3e-4
validation:          every 100 updates
latest checkpoint:   every 100 updates
best checkpoint:     whenever validation loss improves
```

Do not add a scheduler or tune hyperparameters during this run unless there is a clear engineering defect requiring correction.

### Full-data proof pass conditions

The architecture/training system is provisionally proven if:

- training loss falls steadily;
- best validation loss is at least approximately 5% below the step-zero validation loss;
- validation top-1 improves over initialization;
- validation top-5 improves over initialization;
- validation top-16 improves over initialization;
- the improvement is visible across multiple validation checkpoints rather than one isolated lucky point;
- learning is visible across the major turn/shortlist groups rather than only a pathological narrow slice;
- the best checkpoint can be reloaded independently;
- reloaded validation metrics reproduce the saved result within expected numerical tolerance;
- no split contamination is detected.

If training improves strongly while held-out validation remains flat or worsens, do not declare success merely because optimization works. Record the result as an architecture/generalisation failure to investigate.

---

## 9. Evaluate complete validation games

Using the independently reloaded **best checkpoint**, play all **100 validation solutions** through `wordle-ml_game-engine`.

At each move:

- construct state through the same production encoder used for training;
- apply the legal-action mask;
- greedily select the highest-scoring legal action;
- update the game using the authoritative game engine;
- continue until solved or the six-guess limit is reached.

Record:

- solved fraction;
- average guesses;
- guess-count distribution;
- failed solutions;
- full guess/feedback trajectory for every game;
- attempted invalid actions, if any;
- attempted repeated guesses, if any;
- candidate-shortlist size after each move.

Compare these results with the untrained baseline.

The trained checkpoint should improve either solved fraction or average guesses relative to initialization.

Do not expect final-quality Wordle performance from a 2,000-update proof run.

The more important requirement is that the complete training -> checkpoint -> inference -> game-engine path works correctly.

---

## 10. Run cheap architecture-ablation checks

Do not retrain.

Using the best checkpoint, rerun validation inference with each of the following ablations.

### A. Remove candidate-state information

Replace the actual candidate mask and candidate statistics with the equivalent opening/all-solutions representation.

The result should materially worsen at least one important validation metric.

If it does not, investigate whether the model is ignoring the candidate-state branch or whether those tensors are accidentally constant/incorrect.

### B. Remove turn information

Set turn information to one fixed value.

Record the effect, but do not treat a small effect as an automatic failure. Candidate-state information may already imply much of the useful turn information.

### C. Remove candidate bonus

If the architecture uses `remainingActionMask` plus learned `beta`, disable that bonus for evaluation.

Record the effect and correlate it with the learned `beta` telemetry.

This is diagnostic rather than a hard gate.

---

## 11. Produce a concise proof-run report

At the end, create a Markdown report in the machine-learning repository summarising all four stages:

1. untrained baseline;
2. one-batch overfit;
3. mini-dataset run plus resume test;
4. short full-data run.

Include a compact table with, where applicable:

- updates;
- training loss;
- validation loss;
- train top-1/top-5/top-16;
- validation top-1/top-5/top-16;
- validation game win rate;
- validation average guesses;
- checkpoint used;
- pass/fail.

Also include:

- TensorBoard log location and launch command;
- run directory location;
- best checkpoint location;
- relevant Git commit hashes;
- any deviations from this plan;
- any warnings or suspicious behaviour observed.

Do not hide failed gates.

If a gate fails, say clearly where the sequence stopped and why.

---

## 12. Definition of done

This task is complete when all of the following are true:

- single-batch overfitting passed;
- mini-dataset learning passed;
- checkpoint stop/resume was demonstrated;
- the short full-data run produced clear held-out validation improvement;
- TensorBoard contains useful and continuous telemetry;
- the best checkpoint reloads independently;
- the restored model can play all validation games through the shared game engine without integration errors;
- candidate-state ablation demonstrates that the model is using actual game-state information;
- all relevant commits, hashes, configs and metrics are retained;
- a concise Markdown proof-run report has been committed.

Do **not** proceed from here into:

- a long production supervised run;
- extensive hyperparameter tuning;
- architecture redesign;
- soft-target training;
- reinforcement learning.

Stop after the proof report and return the results for review.
