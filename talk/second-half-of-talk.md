# Second Half of the Talk: From GoMLX to cgo and CUDA

## Purpose of this section

This section begins after the audience has already seen:

- how Wordle becomes a state-and-action problem;
- the Go teacher and synthetic training data;
- the small policy model;
- supervised training with GoMLX;
- the successful gameplay result.

At that point, the model is no longer the question. The model works. The second half asks:

> **What actually has to happen for a Go web application to run that trained model directly in hand-written CUDA?**

The narrative should be a controlled descent through the abstraction layers:

```text
working Wordle model
        ↓
Go web application
        ↓
one cgo call
        ↓
C ABI and persistent CUDA state
        ↓
kernel launch
        ↓
blocks, warps and threads
        ↓
registers, shared memory and global memory
        ↓
Nsight Systems and Nsight Compute
        ↓
back to the working application
```

The audience should leave understanding both the overall boundary and one concrete CUDA kernel. They do not need a general CUDA programming course.

## Time budget

The existing talk plan leaves roughly **19 minutes** for this material:

| Part | Target time |
| --- | ---: |
| Pivot from training to explicit inference | 1:30 |
| Combined Go/CUDA application architecture | 1:30 |
| The cgo boundary and ownership model | 2:30 |
| Turning the model into named CUDA kernels | 1:30 |
| Threads, warps, blocks and grids | 3:30 |
| GPU memory, using this implementation | 2:00 |
| Nsight Systems | 2:00 |
| Nsight Compute | 2:00 |
| Correctness, performance and trade-offs | 1:30 |
| Browser demo and conclusion | 1:30 |
| **Total** | **19:30** |

This is already tight. The GPU explanations need to use the Wordle implementation as the example throughout rather than becoming a separate generic lecture.

---

# Proposed slide sequence

## Slide 1: “It learned. Now remove the framework.”

### Purpose

Make a clean transition from the successful machine-learning result into the systems part of the talk.

### Visual

Reuse the complete project diagram from earlier in the talk, but dim everything except:

```text
trained checkpoint → inference → Go application
```

Then replace the GoMLX inference box with:

```text
Go → cgo → C ABI → CUDA
```

### Main points

- Training and inference are different engineering problems.
- GoMLX was the fastest and safest way to prove that the model and training approach worked.
- Once training had produced useful weights, inference was a small, fixed numerical program.
- The CUDA work did **not** replace the successful training path.
- It replaced only the inference backend used by the application.

### Suggested words

> “At this point I had a model that could play Wordle. I did not need to rewrite training. I needed to take this fixed forward pass, peel away the framework, and make the boundary between Go and the GPU explicit.”

### Important honesty point

Say plainly that the model was trained with GoMLX. Do not allow the audience to infer that the successful backpropagation run used the later hand-written CUDA implementation.

---

## Slide 2: “The same model becomes a boring file of numbers”

### Purpose

Bridge from the GoMLX checkpoint to the framework-free runtime without spending time on file-format detail.

### Visual

```text
GoMLX checkpoint
      │
      │ offline exporter
      ▼
manifest.json + FP32 weights + golden test vectors
      │
      ▼
Go/cgo/CUDA application
```

### Main points

- The exporter is allowed to understand GoMLX checkpoints.
- The serving application does not import GoMLX, PJRT or XLA.
- The exporter transposes and writes the trained matrices in one documented, CUDA-friendly order.
- The weights are loaded once and remain resident in GPU memory.
- Golden positions prove that the exported model and the CUDA implementation make the same decisions as the original model.

### Keep this brief

Do not tour the manifest or tensor offsets. The useful conceptual point is:

> **A trained neural network is ultimately weights plus a known sequence of operations.**

### Information to insert from the Codex result

- Exported model size: `[INSERT ACTUAL SIZE]`
- Parameter count: `[INSERT VERIFIED PARAMETER COUNT]`
- Golden-vector action agreement: `[INSERT RESULT]`
- Full-game parity result: `[INSERT RESULT]`

If the implementation differs from the planned neutral export, alter this slide to match the real code.

---

## Slide 3: “One process, two jobs”

### Purpose

Show the complete combined application before looking at code.

### Diagram

```text
Browser
   │ HTTP/JSON
   ▼
Go net/http server
   │
   ├── Wordle game engine
   ├── state encoder
   ├── legal-action mask
   ├── deterministic selection
   │
   ▼
Dedicated Go GPU worker
   │ one synchronous cgo call
   ▼
C ABI and persistent model handle
   │
   ▼
CUDA stream, weights, buffers and kernels
   │
   ▼
4,739 raw logits
   │
   ▼
Go selects the next legal word
```

### Main point

Use the phrase:

> **Go is the control plane; CUDA is the numerical data plane.**

Go still owns everything that is naturally application logic:

- HTTP;
- game state;
- shortlist construction;
- vocabulary identity;
- legality rules;
- repeated-guess suppression;
- deterministic tie-breaking;
- error handling and shutdown.

CUDA receives arrays, evaluates the neural network and returns arrays.

### Why this boundary is good

- CUDA does not need to know Wordle’s rules.
- The web server does not need to know how a dot-product reduction is implemented.
- The existing tested Go game logic is reused rather than duplicated in C++.
- The boundary is narrow enough to explain and test.

### Optional point

The GPU model is owned by one dedicated goroutine locked to an OS thread. Mention this as an ownership decision, not as a deep runtime lesson:

> “One goroutine owns the CUDA stream and model handle, so HTTP handlers cannot race on GPU scratch buffers.”

---

## Slide 4: “The whole forward pass crosses cgo once”

### Required screenshot 1: the real Go cgo code

The screenshot should contain, in the same crop if possible:

```go
/*
#cgo CFLAGS: ...
#cgo LDFLAGS: ...
#include "wordle_cuda.h"
*/
import "C"
```

and:

```go
C.wordle_cuda_model_infer(...)
runtime.KeepAlive(...)
```

Use the final line ranges from the actual implementation. Do not rewrite prettier fake code for the slide.

### Main points to annotate on the screenshot

1. `#cgo` tells the Go linker where the C header and CUDA-built library live.
2. `import "C"` enables the C bridge.
3. Go passes flat FP32 input and output slices.
4. One C call performs the entire forward pass.
5. `runtime.KeepAlive` makes the required Go lifetime explicit.
6. The call returns only after CUDA has copied the result back.

### Explain why one call matters

Bad boundary:

```text
Go → cgo dense layer → Go → cgo ReLU → Go → cgo dense layer → ...
```

Chosen boundary:

```text
Go → cgo complete model inference → Go
```

The chosen design:

- avoids repeated language-boundary crossings;
- keeps CUDA stream ordering inside the native implementation;
- gives the profiler one coherent inference region;
- makes pointer lifetime much easier to reason about.

### cgo ownership explanation

Keep this practical:

- Go owns the input and output slices.
- C may use those addresses only while the synchronous cgo call is active.
- C does not retain a pointer into Go memory.
- The long-lived model handle, weights, stream and scratch buffers are C/CUDA-owned.

### Suggested words

> “The dangerous design would be to launch asynchronous GPU work and let C retain these Go pointers. I did the opposite: the native call is synchronous, and all long-lived state belongs to C and CUDA.”

Do not turn this into a complete presentation on every cgo pointer rule.

---

## Slide 5: “The C ABI hides CUDA C++”

### Required screenshot 2: the real native boundary

Show either the header plus one launch, or two small adjacent crops.

The useful parts are:

```c
typedef struct wordle_cuda_model wordle_cuda_model;

int wordle_cuda_model_infer(...);
```

and a real launch resembling:

```cpp
policy_logits_with_bonus<<<
    kNumActions,
    kDenseThreads,
    0,
    model->stream
>>>(...);
```

### Main points

- CUDA source is compiled as C++, but the public ABI is plain C.
- Go sees an opaque model handle rather than a C++ object.
- The handle owns:
  - weights in device global memory;
  - persistent input and activation buffers;
  - an output buffer;
  - one CUDA stream;
  - error state and device metadata.
- Creation performs allocations and uploads weights.
- Inference reuses the same allocations.
- Destruction releases them.
- There should be no `cudaMalloc` or `cudaFree` in the hot path.

### Good summary line

> “cgo speaks C, so C is the stable front door even when CUDA C++ is behind it.”

---

## Slide 6: “The neural network became seven kernels”

### Purpose

Reconnect the CUDA implementation to the model diagram the audience saw earlier.

### Visual

Use the same model architecture diagram, but label each operation with the corresponding final kernel name from the actual code.

Expected sequence, subject to checking the implementation:

```text
candidate_projection_relu
stats_projection_relu
load_turn_embedding
residual_in_relu
residual_out_skip_relu
candidate_bonus
policy_logits_with_bonus
```

### Main points

- This is the same forward pass and the same trained weights.
- There is no softmax because Go only needs to rank the raw logits.
- The final CUDA output contains one score for every allowed guess.
- Go applies the genuine legal-action mask afterwards.
- The explicit kernel names are useful in Nsight as well as in the code.

### Avoid overexplaining

The audience has already seen the model architecture. This slide exists to establish that each familiar model stage became a concrete GPU operation.

### Useful transition

> “Now we can take one of those kernel launches apart and finally explain what threads, warps and blocks mean.”

---

## Slide 7: “A kernel launch has a shape”

### Use the final policy kernel

This is the best teaching example because the model produces one logit for every possible guess.

Use the real launch geometry. The intended design was:

```cpp
policy_logits_with_bonus<<<4739, 128>>>(...);
```

Verify those numbers against the final code and profiler report before making the slide.

### Explain the syntax

```text
4,739 blocks in the grid
128 threads in each block
32 threads in one warp
4 warps in each block
```

Each block computes one output logit:

```text
block 0    → score for action 0
block 1    → score for action 1
...
block 4738 → score for action 4738
```

### Definitions for Go developers

#### Thread

One logical execution lane running the kernel code. Here, one thread multiplies and accumulates a subset of the 160 input values.

Do **not** call it a goroutine. A CUDA thread is vastly lighter and is executed as part of a warp.

#### Warp

A hardware scheduling group of 32 threads. The lanes usually execute the same instruction together.

For a 128-thread block:

```text
warp 0: threads   0–31
warp 1: threads  32–63
warp 2: threads  64–95
warp 3: threads 96–127
```

#### Block

A group of threads that can cooperate, synchronise and use the same shared memory. One block is assigned to one Streaming Multiprocessor at a time.

#### Grid

All the blocks created by one kernel launch.

### Crucial clarification

The launch describes **logical work**, not the number of threads physically active at once.

```text
4,739 × 128 = 606,592 logical CUDA threads
```

The GPU schedules blocks onto its finite number of Streaming Multiprocessors in waves.

### Suggested words

> “This does not create six hundred thousand CPU threads. It describes six hundred thousand tiny pieces of work, and the GPU schedules them across the hardware it actually has.”

---

## Slide 8: “One block cooperates to calculate one word score”

### Diagram

```text
                     one output weight row
                  160 weights + 160 activations
                                │
                                ▼
┌────────────────────────────────────────────────────────┐
│ block: 128 threads                                     │
│                                                        │
│ warp 0: partial products → subtotal ┐                  │
│ warp 1: partial products → subtotal ├─ shared memory ┐ │
│ warp 2: partial products → subtotal ┤                 ├─ final sum
│ warp 3: partial products → subtotal ┘                 │ │
│                                                        │
│ thread 0: add bias + candidate bonus → write logit ◄──┘ │
└────────────────────────────────────────────────────────┘
```

### Step-by-step explanation

1. Threads take strided portions of the dot product.
2. Each thread accumulates into a register.
3. The 32 lanes in each warp reduce their values using warp shuffle instructions.
4. Four warp leaders write four subtotals into block shared memory.
5. The first warp reduces those four values.
6. One lane adds the bias and candidate bonus and writes the logit.

### Why this example works pedagogically

It gives every CUDA term a visible job:

- **thread:** calculates part of the dot product;
- **warp:** combines 32 thread results;
- **block:** combines four warp results;
- **grid:** produces all possible word scores;
- **kernel launch:** starts that whole operation.

### Branch divergence

Only mention divergence briefly if it naturally follows from the code:

> “Warp lanes are most efficient when they follow the same instructions. When a branch splits a warp, the paths are generally executed separately with different lanes masked off.”

Do not make divergence a major tangent unless Nsight showed it to be relevant.

---

## Slide 9: “GPU memory has scope”

### Visual

Use a simple vertical journey rather than a generic memory-hierarchy pyramid:

```text
Go host memory
  candidate mask, statistics, action mask
          │ HtoD copies
          ▼
Device global memory
  weights, inputs, activations, logits
          │
          ├── cached automatically in L2/L1 where useful
          │
          ▼
Registers
  one thread's running accumulator
          │ warp reductions
          ▼
Shared memory
  four warp subtotals visible to one block
          │
          ▼
Device global memory
  4,739 output logits
          │ DtoH copy
          ▼
Go host memory
```

### Memory types to explain

| Memory | Scope | Use in this inference |
| --- | --- | --- |
| Host memory | CPU/Go | Input state and returned logits |
| Device global memory | All GPU threads | Weights, input arrays, activations and output |
| L2/L1 cache | Hardware-managed | Repeated reads of activations and weights |
| Registers | One CUDA thread | Partial dot-product accumulator |
| Shared memory | One CUDA block | The four warp-level subtotals |
| Constant memory | Device-wide, read-only | Not necessary for the main computation |
| Local memory | Logically private, often backed by device memory | Compiler spills; something to avoid rather than celebrate |

### Two memorable lines

> “Shared memory is shared by a block, not by the whole GPU.”

> “CUDA local memory sounds fast, but it often means a register spill into much slower device memory.”

### Keep the emphasis on scope

The key lesson is not simply “some memory is faster”. It is:

> Faster memory is smaller, and it is visible to a narrower part of the program.

---

## Slide 10: “Nsight Systems: the whole request”

### Required screenshot 3: NVIDIA Nsight Systems

Use a saved report rather than profiling live on stage.

The screenshot should show one complete inference and, where available:

- the dedicated `wordle-gpu` host thread;
- the outer NVTX inference range;
- input host-to-device copies;
- all seven named kernel launches;
- the logits device-to-host copy;
- the final synchronisation;
- enough of the CPU row to connect the work to the cgo call.

Hide unrelated processes and rows. Zoom tightly enough that the operations and names are readable.

### Explain the product in one sentence

> **Nsight Systems is the wide-angle lens: it shows how CPU work, CUDA API calls, memory transfers, kernels and synchronisation fit together over time.**

### How to narrate the screenshot

Move left to right:

1. Go enters the native call.
2. Inputs are copied to device memory.
3. The seven kernels execute in stream order.
4. The logits are copied back.
5. The call synchronises and Go resumes.

### Use it to answer a question

Do not merely identify coloured rectangles. State the actual finding from the collected trace:

- total inference range: `[INSERT ACTUAL VALUE]`;
- time or proportion in copies: `[INSERT ACTUAL VALUE]`;
- time or proportion in kernels: `[INSERT ACTUAL VALUE]`;
- visible launch/synchronisation overhead: `[INSERT ACTUAL OBSERVATION]`;
- any first-call warm-up effect: `[INSERT ACTUAL OBSERVATION]`.

A likely result is that this batch-one Wordle inference is so small that transfer and launch overhead are important. Only say this if the trace supports it.

### Speaker warning

Nsight Systems is visually dense. Add two or three callout boxes or arrows to the slide and ignore everything else.

---

## Slide 11: “Nsight Compute: inside one kernel”

### Required screenshot 4: NVIDIA Nsight Compute

Profile the final policy kernel, expected to be named:

```text
policy_logits_with_bonus
```

The screenshot should display a useful combination of:

- kernel name;
- grid dimensions;
- block dimensions;
- warps per block;
- registers per thread;
- shared memory per block;
- achieved occupancy;
- memory workload or throughput;
- source correlation around the multiply-accumulate and reduction.

### Explain the product in one sentence

> **Nsight Compute is the microscope: it collects detailed measurements for one CUDA kernel.**

### Connect it to the previous slides

Point out that the profiler independently reports the launch shape just explained:

```text
4,739 blocks
128 threads per block
4 warps per block
```

Then connect reported resource use to the memory slide:

- registers hold per-thread accumulators;
- shared memory holds warp subtotals;
- global loads fetch weights and activations;
- resource use affects how many blocks can be resident on an SM.

### Use it to answer one concrete question

Fill this in from the actual report:

- Is the kernel primarily limited by memory traffic, arithmetic throughput, latency or some other factor?
- Are register or shared-memory requirements limiting occupancy?
- Does the source view show any spills or unexpectedly expensive instructions?
- Is there enough work per launch to use the GPU effectively?

Write the final slide conclusion as a single sentence:

> `[INSERT THE ACTUAL PROFILER-SUPPORTED CONCLUSION]`

### Important caveat

Do not present occupancy as a score where 100% automatically means “fast”. It is one constraint and diagnostic among several.

---

## Slide 12: “Correct first; faster only if measured”

### Purpose

Show that the CUDA backend is the same model before discussing timing.

### Suggested table

Replace every placeholder with the actual Codex report values.

| Check | Result |
| --- | --- |
| Exported parameters | `[VALUE]` |
| Golden positions | `[COUNT]` |
| Maximum logit difference | `[VALUE]` |
| Top-1 action agreement | `[VALUE]` |
| Top-5 agreement | `[VALUE]` |
| Fixed-game trajectory agreement | `[VALUE]` |
| Validation games solved | `[VALUE]` |
| Mean guesses | `[VALUE]` |
| Illegal selections | `[VALUE]` |

Then show the latency measurements separately:

| Backend or measurement | Warm median | p95 | Notes |
| --- | ---: | ---: | --- |
| GoMLX inference | `[VALUE]` | `[VALUE]` | `[CONDITIONS]` |
| Hand-written CUDA/cgo | `[VALUE]` | `[VALUE]` | `[CONDITIONS]` |
| First CUDA call | `[VALUE]` | — | Includes warm-up if applicable |

### Main message

Do not force a speedup story.

A small batch-one network can be dominated by:

- host/device copies;
- kernel-launch latency;
- synchronisation;
- cgo and host orchestration;
- poor arithmetic intensity in matrix-vector operations.

The engineering achievement is still real:

- no ML framework in the serving binary;
- explicit ownership;
- known memory movement;
- inspectable kernels;
- numerical parity;
- the ability to profile and optimise from evidence.

### Suggested words

> “CUDA does not make every computation faster by existing near it. For this tiny single-state inference, the important result was control and visibility. The profiler tells us what would have to change before speed improved—probably batching, fewer launches, or less copying.”

Only mention specific future optimisations that follow from the actual profiles. Candidates include:

- batching simultaneous games;
- CUDA Graphs to reduce repeated launch overhead;
- pinned staging buffers;
- compact input masks;
- returning only the most useful outputs;
- fusing kernels;
- lower precision after an FP32 correctness baseline.

Do not suggest that these were implemented unless they were.

---

## Slide 13: “Back to the application”

### Required screenshot or live demo 5: browser application

The browser should visibly show:

```text
Backend: CUDA via cgo
Model: <run/checkpoint/update>
Device: <GPU name>
```

Play or show one complete game.

### What to say

- It is still an ordinary Go web application.
- The browser and HTTP handlers do not know CUDA details.
- The Go game engine constructs state and enforces the rules.
- Each turn makes one complete inference call.
- CUDA returns scores; Go chooses and applies the legal word.

### Demo strategy

Prefer one of these, in order:

1. a very short live game with the server already running;
2. a prepared browser recording embedded in the slides;
3. a sequence of screenshots showing one complete game.

Do not compile CUDA, export weights or collect an Nsight report live on stage.

### Link back to the opening

If the talk opened with the model playing, show the same visual again and reveal that the audience now understands the complete route behind each guess.

---

## Slide 14: “What Go did, and what CUDA did”

### Visual

Return to the original complete project diagram for the final time. Highlight the entire path:

```text
Go teacher
   → synthetic data
   → GoMLX training on the GPU
   → exported weights
   → Go web application
   → cgo/CUDA inference
   → Wordle move
```

### Three conclusions

1. **Go was the system language.**  
   It handled game logic, data generation, reproducibility, validation, orchestration, serving and the user-facing application.

2. **GoMLX was the quickest route to a model that actually learned.**  
   It made backpropagation and experimentation practical without preventing a lower-level implementation later.

3. **cgo and CUDA made the numerical boundary explicit.**  
   They allowed the same trained policy to run through code whose memory movement, launch structure and hardware behaviour could be seen directly.

### Suggested closing words

> “The point was not to replace Go with CUDA. The point was to give each side the work it is good at, and to make the boundary between them small enough that we could understand it.”

Then finish on the model solving Wordle, not on a profiler table.

---

# Diagrams and assets still needed

The second half needs a small, coherent asset set. Reuse shapes and colours so the audience sees one system being progressively expanded rather than a collection of unrelated diagrams.

## Essential diagrams

### 1. Combined application boundary

```text
browser → Go application → GPU worker → cgo → C ABI → CUDA → logits → Go rules
```

This should be the main map for the second half.

### 2. Model-to-kernel diagram

Reuse the first-half model architecture and add the final CUDA kernel name to each operation.

### 3. One-logit-per-block diagram

Show four warps reducing into shared memory and producing one output score.

### 4. Actual memory journey

Use this application’s inputs, weights, accumulators and logits rather than an abstract GPU memory pyramid.

## Essential captured assets

- Real Go cgo screenshot.
- Real C ABI/CUDA launch screenshot.
- Nsight Systems screenshot.
- Nsight Compute screenshot.
- Browser CUDA-backend screenshot or recording.
- Correctness/parity report values.
- Benchmark values with clearly documented conditions.

## Codex-output extraction checklist

Before finalising slides, inspect the Codex completion report and repository and fill in:

- branch and commit containing the implementation;
- exact command that starts the combined web application;
- actual cgo source file and line range;
- actual header and `.cu` source file and line range;
- real exported model path and size;
- exact model parameter count;
- exact kernel names and launch dimensions;
- actual number and direction of memory copies;
- golden-vector parity metrics;
- fixed-game parity metrics;
- benchmark median and p95;
- first-call versus warm-call timing;
- Nsight Systems report and screenshot path;
- Nsight Compute report and screenshot path;
- the profiler-supported performance conclusion;
- browser URL and backend metadata shown in the UI.

Do not leave placeholders in the final talk.

---

# What the audience must understand

By the end of this section, a Go developer with no previous GPU experience should be able to answer:

1. Why is there one cgo call rather than one call per layer?
2. Which parts of the application remain in Go?
3. Who owns the trained weights and GPU buffers?
4. What is a CUDA kernel launch?
5. What is the difference between a thread, warp, block and grid?
6. How does one block calculate one Wordle action score?
7. What lives in host memory, global device memory, registers and shared memory?
8. What different questions do Nsight Systems and Nsight Compute answer?
9. Why is a GPU implementation not automatically faster for a tiny batch-one model?
10. How was correctness checked against the original GoMLX model?

If an explanation does not help answer one of those questions, it probably does not belong in the main talk.

---

# Things not to say

Avoid these misleading shortcuts:

- “We trained the model in hand-written CUDA.”
- “A CUDA thread is basically a goroutine.”
- “All 606,592 threads run at once.”
- “Threads in different blocks can share ordinary shared memory.”
- “Local memory is the fast memory closest to a thread.”
- “High occupancy means the kernel is fast.”
- “cgo is slow, therefore the boundary must be expensive.”
- “CUDA made it faster” without comparable measurements.
- “The GPU chooses a legal Wordle move.”
- “The whole application is now in CUDA.”
- “The Nsight screenshot proves it is optimised.”

More accurate alternatives are already given in the slide notes above.

---

# Material to keep in backup slides

Useful backup material for questions, but too detailed for the main 19 minutes:

- complete C ABI;
- exact tensor names, dimensions and weight offsets;
- neutral model manifest;
- pure-Go reference evaluator;
- complete golden-vector parity table;
- full 100-game trajectory comparison;
- cgo pointer rules and why `runtime.KeepAlive` is present;
- why the GPU worker calls `runtime.LockOSThread`;
- error handling across the C ABI;
- build process from `.cu` to PIC object to static library to cgo-linked binary;
- complete seven-kernel launch table;
- full Nsight Systems timeline;
- additional Nsight Compute sections;
- proposed future optimisation order;
- the unsuccessful reinforcement-learning experiment, only if someone asks whether it was tried.

Do not place the RL experiment in the main narrative. The successful line is cleaner:

```text
Go teacher → supervised imitation → working model → explicit CUDA inference
```

---

# Overrun plan

If the first half runs long, preserve these elements:

1. explicit transition from GoMLX training to hand-written CUDA inference;
2. combined Go/CUDA architecture;
3. real cgo screenshot;
4. one-logit-per-block explanation of threads, warps and blocks;
5. one memory slide;
6. Nsight Systems screenshot;
7. Nsight Compute screenshot;
8. final application and conclusions.

Cut or compress in this order:

1. neutral export-format details;
2. detailed parity table—state the result verbally;
3. additional cgo ownership details;
4. branch-divergence aside;
5. future-optimisation list;
6. live demo—replace it with the prepared recording.

Do not cut either Nsight product, because the talk specifically needs both perspectives.

---

# Rehearsal checklist

Before considering the second half ready, be able to explain without notes:

- the entire Go-to-CUDA request path in 30 seconds;
- why training used GoMLX but serving does not;
- why one cgo call covers the whole forward pass;
- why C does not retain Go pointers;
- what the persistent native handle owns;
- the actual final policy-kernel launch dimensions;
- the roles of a thread, warp, block and grid;
- how the block reduction works;
- where the accumulator, weights, warp subtotals and logits live;
- the exact difference between Nsight Systems and Nsight Compute;
- the actual measured correctness result;
- the actual warm inference latency and measurement conditions;
- the one profiler-supported performance conclusion;
- why the project is still fundamentally a Go application.

Also rehearse the transitions:

### Into the second half

> “We have proved that the model can learn. Now let us take away the framework and follow one guess all the way from Go into the GPU.”

### Into the CUDA execution model

> “This one launch is enough to explain most of the vocabulary: a grid of blocks, each block made of warps, and each warp made of 32 threads.”

### Into profiling

> “We have a mental model of what should happen. Now the profiler can tell us what actually happened.”

### Into the conclusion

> “After all that, the browser still sees an ordinary Go server. The complexity is contained behind one deliberate boundary.”

---

# Condensed narrative

The whole second half can be summarised as:

> The model was trained successfully with GoMLX. We exported its weights, retained Wordle and web logic in Go, and put the fixed numerical forward pass behind one synchronous cgo call. A persistent C/CUDA model handle owns the device resources. Seven named kernels evaluate the network. In the final policy kernel, one 128-thread block—four warps—cooperates to calculate each of 4,739 possible word scores using registers, warp reductions and block shared memory. Nsight Systems shows the complete Go-to-GPU request; Nsight Compute explains one kernel in detail. The result is still a Go application, but with a GPU boundary we can see, measure and understand.
