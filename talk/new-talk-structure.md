# New Talk Structure

## Core shape

The talk should be **linear**, not tree-like. Use a tree or pipeline on the contents slide, then follow one clear route:

**Wordle → Go teacher → synthetic data → model → backprop on the GPU → explicit Go/CUDA boundary → working application**

There is one important contract issue: **do not imply that the successful training run used hand-written CUDA or cgo.** Make the evolution of the project part of the story:

> “I originally intended to write the training path directly in CUDA. To get a real model working, I first used GoMLX. Once it worked, I could peel back the framework and implement the inference boundary explicitly.”

That is honest, still substantially on brief, and gives GoMLX a legitimate role rather than treating it as an omission.

## Proposed running order — 55 minutes plus 5 minutes for questions

1. **Contents and central idea — 2 minutes**
   - Show the complete pipeline.
   - Establish the division of labour:
     - Go handles Wordle, data, orchestration, validation and serving.
     - The GPU handles the dense numerical work.
   - Promise that the talk will finish by looking directly at the Go/CUDA boundary.

2. **Wordle in ninety seconds — 4 minutes**
   - Explain guesses, feedback colours, six-turn limit and candidate words.
   - Show a very brief example game.
   - Ideally show the trained model playing immediately, so the audience knows where the story is going.

3. **The complete project architecture — 3 minutes**
   - One visual containing:
     - Go game engine
     - Go teacher
     - synthetic-data generator
     - GoMLX model and trainer
     - CUDA GPU
     - Go HTTP server and browser demo
   - This becomes the map you revisit as the talk progresses.

4. **The Go program that knows how to play Wordle — 7 minutes**
   - Explain the candidate shortlist.
   - Explain the teacher’s rule: evaluate every unused guess by its **worst-case remaining shortlist**, while still allowing information-gathering probe words.
   - Show one useful Go code excerpt rather than touring the repository.
   - Use concurrency as the Go-specific engineering example:
     - precompute feedback for guess/solution pairs;
     - generate records for different solutions through a fixed worker pool;
     - collect deterministic output despite parallel execution.
   - A small interactive animation of guesses being distributed among workers would suit the slide system well.

5. **Turning a solver into training data — 4 minutes**
   - Do not discuss CSV or binary layouts in detail.
   - Visualise **one training example**:
     - current candidate set;
     - turn number;
     - candidate statistics;
     - teacher’s preferred next guesses.
   - Explain that the generator creates incomplete states at different depths and stores the teacher’s preferred choices.
   - Mention that train/validation/test splits are separated appropriately.
   - Main lesson: synthetic data lets a slow, deliberative Go program teach a much cheaper policy.

6. **The model architecture — 8 minutes**
   - This should be the main technical diagram and the part to rehearse most carefully.
   - Animate the model inputs flowing through the network:
     - candidate information;
     - candidate statistics;
     - turn/depth information;
     - learned projections/embeddings;
     - concatenated internal representation;
     - compact residual/MLP trunk;
     - action logits over possible guesses.
   - Explain the purpose of each branch, not every matrix dimension.
   - Emphasise the simplicity: roughly a million parameters, no attention, no giant language model.
   - Make sure the exact current architecture is understood and represented accurately before finalising these slides.

7. **Training it with backpropagation and GoMLX — 8 minutes**
   - Briefly explain supervised imitation:
     - state goes in;
     - teacher’s preferred action is the target;
     - loss measures the error;
     - backpropagation adjusts the weights.
   - Show a compact GoMLX code or graph-building excerpt.
   - Show one TensorBoard training curve and perhaps the checkpoint/evaluation loop.
   - Lead with the successful result: the model went from essentially not playing Wordle to solving almost the entire fixed evaluation set, with a low mean guess count.
   - This is the emotional high point of the project: the small policy really learned to play.

8. **Peeling back GoMLX: an explicit Go/cgo/CUDA inference path — 10 minutes**
   - This is the only substantial additional development worth attempting.
   - Give a brief, just-in-time GPU recap:
     - host versus device memory;
     - kernels;
     - threads, warps and blocks;
     - launch and transfer overhead.
   - Then follow one inference call:
     - Go encodes the state;
     - cgo enters a stable C ABI;
     - trained weights remain resident in GPU memory;
     - CUDA performs the model’s numerical operations;
     - logits return to Go;
     - Go applies policy constraints and selects a word.
   - Compare the hand-written path with GoMLX:
     - same trained checkpoint;
     - same inputs;
     - numerically equivalent results;
     - very different levels of abstraction.
   - Do **not** try to rewrite the whole training system in CUDA now.

9. **Profiling the boundary with NVIDIA Nsight — 5 minutes**
   - Use **Nsight Systems** for one timeline showing:
     - Go execution;
     - cgo call;
     - host/device transfers if present;
     - kernel launches;
     - synchronisation.
   - Use **Nsight Compute** for at most one kernel screenshot.
   - Use the screenshots to answer a concrete engineering question rather than merely proving that Nsight was opened.
   - A likely useful lesson is that a single Wordle state is tiny, so launch and transfer overhead may matter more than raw arithmetic throughput.

10. **Return to the application and conclude — 4 minutes**
    - Show the browser game or a recorded fallback.
    - Return to the original system diagram, now fully explained.
    - Finish with three lessons:
      - Go is excellent for constructing and operating the surrounding system.
      - GoMLX was the fastest route to proving that the model and backpropagation worked.
      - cgo and CUDA let you take control where the abstraction boundary becomes technically interesting.
    - End on the trained model solving a Wordle rather than on a wall of benchmark numbers.

## What to leave out

- **No reinforcement-learning section.**
  - It is not in the abstract.
  - It did not improve the result.
  - It interrupts the cleaner teacher → imitation → successful model story.
  - Keep one hidden backup slide only if useful for questions.
- No detailed file-format tour.
- No long generic GPU-architecture lecture repeating last year’s talk.
- No derivation of backpropagation equations.
- No implication that the successful training checkpoint was produced by the hand-written CUDA/cgo implementation.

## Remaining technical work, in priority order

1. **Build and understand the model diagram.**
   - Mandatory even if no more code is written.
   - Verify the exact current architecture against the implementation.

2. **Ask Codex for a bounded CUDA inference backend.**
   - Export/load the existing trained weights.
   - Load weights once and keep them resident on the GPU.
   - Expose inference through a small C ABI and cgo wrapper.
   - Prefer support for batched inference.
   - Compare against GoMLX on fixed golden states.
   - Require identical top choices and logits within a stated numerical tolerance.

3. **Profile the CUDA inference path.**
   - Capture one useful Nsight Systems trace.
   - Capture one useful Nsight Compute view.
   - Turn each screenshot into a specific teaching point.

4. **Optional presentation polish only after the above works.**
   - Add a GoMLX/CUDA backend toggle to the web demo.
   - Add interactive slide widgets or animations around:
     - Go worker concurrency;
     - the model data flow;
     - host/device execution;
     - CUDA threads/warps/blocks.

## Narrative in one sentence

**We start with a Go program that can reason its way through Wordle, use it to manufacture training examples, teach a compact neural network with backpropagation on the GPU, and then peel back the ML framework to see exactly how Go can cross the cgo boundary and run that learned model in CUDA.**
