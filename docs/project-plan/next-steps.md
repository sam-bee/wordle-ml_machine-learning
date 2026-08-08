# Next steps

The policy model, canonical vocabulary and split contract, WDIT v3 teacher
corpus, shared state encoder, strict data reader, deterministic batching, and
inspection command are complete. The old pre-training implementation plan has
been removed from this file because every data-preparation step in it has
landed.

The immediate work is:

1. Review and land the initial supervised-training plumbing: sparse
   cross-entropy against the teacher's top action, an Adam optimizer,
   train/validation metrics, checkpoints, and TensorBoard scalar output.
2. Run a small, bounded training experiment. First prove that the model and
   loss can overfit a small dataset, then train against the frozen training
   records while using only validation records to assess the run.
3. Record the exact command, seed, commit, metrics, checkpoint, and observations
   before deciding whether the model, objective, or data needs to change.

The 100-solution final-test split remains sealed until model and training
decisions are finished. Gameplay, reinforcement learning, and a custom CUDA
model are not part of the first experiment.
