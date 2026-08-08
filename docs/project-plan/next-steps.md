# Wordle ML — Next Steps Before First Training Run

This phase is about preparing the data and training pipeline so that the first supervised training run has a clean, reliable input contract. It deliberately stops **before** any actual training experiments.

## 1. Fix and verify Wordle feedback correctness

Before generating the definitive training data, verify the feedback logic in `wordle-ml_game-engine`, especially repeated-letter handling.

- Correct the feedback implementation so that yellow-letter matches consume available occurrences in the solution.
- Add regression tests covering duplicate letters in both guesses and solutions.
- Treat `wordle-ml_game-engine` as the authoritative implementation of Wordle rules used by:
  - synthetic data generation;
  - shortlist updates;
  - later complete-game evaluation.

This needs to happen first because incorrect feedback would contaminate candidate shortlists and teacher decisions throughout the generated dataset.

## 2. Freeze the vocabulary and dataset split contract

Define one canonical ordering for all IDs used by the model and data generator.

- Use the machine-learning repository's fixed:
  - **2,309 candidate solutions**;
  - **4,739 allowed model actions**.
- Preserve the intended split:
  - **2,109 training solutions**;
  - **100 validation solutions**;
  - **100 final-test solutions**.
- Update `wordle-ml_synthetic-data-creation` so it consumes these explicit vocabularies and splits instead of maintaining its own independent split logic.
- Store vocabulary/version hashes in generated dataset metadata.
- Make the training loader fail loudly if dataset vocabulary dimensions or hashes do not match the model repository.

The important goal is that a word ID means exactly the same thing everywhere.

## 3. Make the synthetic-data project a reusable Go package

Use `wordle-ml_synthetic-data-creation` as the teacher and training-example source rather than copying its logic into the ML repository.

- Give the project a proper importable Go module path if required.
- Expose a small reusable API for generating training states.
- Reuse its existing fast components, including:
  - the precomputed feedback matrix;
  - teacher scoring/ranking;
  - generation of fresh and in-progress games.
- Keep the synthetic generator responsible for producing examples, while the ML repository remains responsible for converting those examples into tensors and training the model.

A generated example should expose enough information to reproduce the model state without rerunning expensive teacher logic.

## 4. Add the complete candidate shortlist to each training example

The model architecture is designed to use the current remaining-solution set as part of its input, so the dataset should contain it explicitly.

For each generated state, store:

- game/turn history;
- current turn number;
- complete remaining-candidate set;
- shortlist size;
- teacher's ranked next moves;
- existing teacher scores/statistics useful for later losses or diagnostics.

Represent the candidate set compactly as a **2,309-bit bitset** (about 289 bytes).

Also add integrity checks such as:

- bitset population count equals the stored shortlist size;
- the true solution is still present in the candidate set;
- all stored word IDs are within the frozen vocabularies.

Version the record format so future additions do not silently break older datasets.

## 5. Build one shared model-state encoder

In `wordle-ml_machine-learning`, implement a single encoder that converts a generated game state into the exact tensors expected by the existing ~1.1M-parameter GoMLX model.

Given the candidate bitset and turn/game state, produce the model inputs, including:

- candidate mask over the 2,309 solutions;
- candidate-derived statistical features expected by the model;
- turn information;
- remaining-action mask over the 4,739 possible guesses.

Create and test the fixed mapping between solution IDs and action IDs.

This encoder should become the canonical representation used by both:

- offline supervised training;
- later live game-playing/evaluation.

Avoid creating separate "training" and "gameplay" encoders.

## 6. Wire the dataset reader and batch pipeline

Add the infrastructure required to feed examples into GoMLX, without yet running a real training experiment.

- Read the versioned binary dataset format.
- Validate metadata and vocabulary hashes on load.
- Decode candidate bitsets and teacher labels.
- Convert examples through the shared model-state encoder.
- Assemble batches with the exact tensor shapes expected by `policy.Model.Forward`.
- Support deterministic shuffling/seeding.
- Keep training, validation and final-test records strictly separated.

Add a small inspection/debug command that can load a record and print enough human-readable information to verify:

- the game history;
- remaining candidate count;
- a sample of remaining candidate words;
- teacher top choices;
- resulting model tensor dimensions.

## 7. Implement the initial supervised objective and training plumbing

Prepare the code needed for the first imitation-learning run, but do not launch it yet.

Start with the simplest objective:

- model logits over the 4,739 allowed guesses;
- illegal/already-used actions masked out;
- cross-entropy against the teacher's top-ranked move.

Also wire the metrics that will be useful once training begins:

- loss;
- teacher top-1 agreement;
- teacher top-5 agreement;
- teacher top-16 agreement.

Keep the teacher's additional ranked moves and scores in the dataset so that a softer top-k objective can be added later without regenerating data, but do not make that complexity a prerequisite for the first run.

Add:

- optimizer construction;
- checkpoint save/load support;
- TensorBoard metric hooks;
- deterministic seeds/configuration;
- a command/configuration entry point for starting training later.

## Definition of done for this phase

Before attempting the first training run, we should be able to:

- generate a valid training example from the reusable teacher package;
- inspect its exact remaining candidate shortlist;
- serialize and deserialize it without information loss;
- convert it into the tensors required by the current GoMLX model;
- pass a batch through the model and compute the supervised loss;
- verify vocabulary, split and tensor-shape invariants automatically;
- initialize the optimizer, metrics and checkpoint machinery.

At that point, the data/model boundary is defined and tested, and the **next** task can be the first deliberately small training experiment.
