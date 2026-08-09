# Hand-written CUDA inference via cgo

This document describes the direct CUDA/cgo demonstration route. It is
intentionally separate from the retained [GoMLX/XLA serving route](inference-serving.md):
the older route can select and warm compatible checkpoints behind an internal
inference service, while `cudaweb` loads one exported model and is one Go
process serving both browser and inference requests.

The goal is a small, inspectable conference demonstration, not a general
tensor library or a replacement for training. GoMLX remains the offline
training, restoration, and reference implementation. The CUDA runtime, pure-Go
portable evaluator, verifier, benchmark, and browser demo do not import
GoMLX, PJRT, or XLA.

## Portable model artifact

`make cuda-cgo-export` restores a completed checkpoint through the supported
GoMLX run/checkpoint validation path, materializes its trainable variables,
validates the vocabulary hashes and fixed model shape, and exports a neutral
artifact. The default selected source is recorded in the
[CUDA/cgo working notes](../project-plan/cuda-cgo-inference-working-notes.md);
the written manifest is authoritative for the actual run, checkpoint, update,
training commit, and hashes.

A successful export is published under the selected run, for example:

```text
runs/<run-id>/exports/cuda-f32-v1/best/
├── manifest.json
├── weights.f32le
├── golden-vectors.json
├── golden-vectors.f32le
├── golden-games.json
└── export-report.json
```

`manifest.json` identifies format `wordle-cuda-f32-v1`, little-endian FP32,
the run and checkpoint, the 2,309-solution / 4,739-action dimensions, 209
candidate statistics, six turns, trunk width 160, the exact 1,046,596
parameters, both vocabulary hashes, and the SHA-256 of `weights.f32le`. Its
tensor table records each stable tensor name, shape, float offset, count, and
source checkpoint variable. Dense matrices are transposed once into
output-major `[outputs, inputs]` rows so one CUDA block reads a contiguous
output row.

The loader rejects an unknown version, unexpected tensor order or shape,
incorrect vocabulary hash, bad FP32 payload size, non-finite weight, or
SHA-256 mismatch before the flat weight slice reaches C. The 4,186,384-byte
payload is therefore a validated, fixed contract rather than an undocumented
checkpoint scrape. The golden sidecars contain representative raw-logit inputs
and selected-action information; `golden-games.json` records the reference
validation-only trajectories. They never contain final-test play.

## Selected export and verification evidence

The completed selected export is
`seed-replication-20260809-132505Z`, checkpoint `best`, update **2,600**. Its
manifest records 1,046,596 parameters and this verified weight payload SHA-256:

```text
b78dc980505998d9dd40551ef4d24788b8378be63e4d09fb90aa0a8be83c870d
```

[`verification-report.json`](../../runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best/verification-report.json)
compares 32 golden vectors (151,648 logits per comparison) at an absolute
tolerance of `1e-3`:

| Comparison | Maximum absolute error | Mean absolute error | Top-1 / top-5 / selected action |
| --- | ---: | ---: | --- |
| GoMLX reference → portable Go | `1.52587890625e-05` | `1.2925607276829653e-06` | 32/32 / 32/32 / 32/32 |
| Portable Go → CUDA | `1.9073486328125e-05` | `1.316824988479718e-06` | 32/32 / 32/32 / 32/32 |
| GoMLX reference → CUDA | `7.62939453125e-06` | `7.5632718075904e-07` | 32/32 / 32/32 / 32/32 |

There were no tolerance failures or documented near ties. The same report
records **100/100 exact validation-game trajectories**, including raw top
action, selected action, accepted guess, shortlist transition, solved state,
and guess count. The matching summary is **98/100 solved** with **3.66 mean
guesses**. This is parity evidence for this fixed validation population and
checkpoint, not a broader generalization claim.

## Ownership boundary

```text
Go HTTP + Wordle engine + vocabulary + modelstate + legal action selection
       │ modelstate.Inputs (four tensors)
       │ one synchronous cgo call
       ▼
C ABI + persistent CUDA handle + one stream
       │ raw FP32 logits[4739]
       ▼
Go availability mask + lower-action-ID tie break + game progression
```

Go is the control plane. It validates and loads vocabularies, checks their
hashes, encodes the board, keeps the authoritative Wordle game, removes
previous guesses with the separate available-action mask, and selects an action
from returned logits. CUDA is only the numerical data plane: it receives
`candidateMask`, `candidateStats`, `turn`, and `remainingActionMask`, then
returns every raw action logit.

`remainingActionMask` is the learned candidate bonus input, not a legality
mask. CUDA does not receive the available-action mask and does not perform
argmax, duplicate suppression, or a softmax. Probe words retain ordinary raw
logits; the existing Go selector decides whether they are available.

The plain C ABI in
[`wordle_cuda.h`](../../wordleml/cuda/inference/wordle_cuda.h) exposes an
opaque `wordle_cuda_model`. At creation, C owns the selected CUDA context
state, one stream, one contiguous device allocation for all 1,046,596 weights,
persistent device input buffers, `h[160]`, residual scratch, scalar `beta`,
and `logits[4739]`. Allocation, weight upload, and device validation happen at
creation; freeing happens at destruction. There is no `cudaMalloc`,
`cudaFree`, stream creation, or stream destruction in a forward call.

`cudainfer` creates that handle in one goroutine after `runtime.LockOSThread`.
That goroutine owns and serializes every native call, destroys the handle before
unlocking, and labels its Linux thread `wordle-gpu` for an Nsight timeline.
HTTP goroutines communicate through typed requests instead of concurrently
calling one CUDA model.

The cgo wrapper passes Go slice data only for the duration of
`wordle_cuda_model_infer`, then calls `runtime.KeepAlive` on all four slices.
C never stores a Go pointer. The native implementation synchronizes its stream
after the logits DtoH copy, and also synchronizes before returning an error
after asynchronous work; Go therefore never returns to a caller while CUDA may
still read caller-owned memory. A cancelled request is rejected before enqueue;
once native inference starts, the worker lets the short synchronous call finish
so this lifetime rule remains true.

## Fixed forward pass

The host sums the candidate mask and passes its reciprocal to the first kernel.
This validates a non-empty candidate set without an eighth normalization
kernel. One inference then enqueues, in one stream:

1. candidate-mask, candidate-statistics, and remaining-action-mask HtoD copies;
2. the seven named kernels below, in graph order;
3. one logits DtoH copy and stream synchronization.

| Kernel | Grid | Block | Work |
| --- | ---: | ---: | --- |
| `candidate_projection_relu` | 96 | 128 | normalized candidate projection and ReLU |
| `stats_projection_relu` | 48 | 128 | statistics projection and ReLU |
| `load_turn_embedding` | 1 | 32 | selected 16-value turn embedding |
| `residual_in_relu` | 160 | 128 | first residual dense layer and ReLU |
| `residual_out_skip_relu` | 160 | 128 | second dense layer, skip addition, and ReLU |
| `candidate_bonus` | 1 | 128 | scalar candidate bonus |
| `policy_logits_with_bonus` | 4,739 | 128 | one raw action logit per block |

Each dense block holds one register subtotal per thread, reduces each warp with
shuffle operations, and uses shared memory only for the four warp-leader
subtotals. Inputs, weights, activations, and logits are FP32; only the turn is
an integer. The native compile uses `-O3`, `-lineinfo`, `-std=c++17`,
`-arch=sm_120`, and PIC. It does not use Tensor Cores, FP16/BF16, TF32-specific
intrinsics, `--use_fast_math`, cuBLAS, CUTLASS, or a CPU implementation.

On creation the backend requires exactly one visible GPU named `NVIDIA GeForce
RTX 5070 Ti`, `NVIDIA GeForce RTX 5050`, or `NVIDIA GeForce RTX 5050 Laptop
GPU`, at compute capability 12.0. It rejects every other device, including the
RTX 3060, and reports device name, compute capability, CUDA runtime version,
and driver version through the UI/API metadata.

## Build, export, verify, benchmark, and demo

All commands run through the project Docker environment and its UUID-selected,
approved GPU. Set `RUN_ID`, `CHECKPOINT`, and `MODEL_DIR` when a different
completed model is intended. `MODEL_DIR` is repository-relative below:

```console
make cuda-cgo-export RUN_ID=seed-replication-20260809-132505Z CHECKPOINT=best
make cuda-cgo-build
make cuda-cgo-test
make cuda-cgo-verify MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
make cuda-cgo-bench MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
make cuda-cgo-demo MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

`cuda-cgo-build` first compiles the `.cu` file with `nvcc` into a PIC object,
archives `build/cuda/libwordle_cuda.a`, then builds the tagged `cudaverify`,
`cudabench`, and `cudaweb` commands with `CGO_ENABLED=1 -tags cuda_cgo`.
cgo does not compile CUDA source directly. Generated native objects, binaries,
model payloads, and binary profiler reports are not committed; the compact
text/CSV interpretations and browser PNG below are tracked talk evidence.

The standalone verifier loads the artifact and its golden sidecars without
GoMLX. It compares the stored GoMLX reference logits with the portable Go
evaluator and CUDA, reports absolute errors/agreement/finite checks and any
action divergence, then compares complete CUDA validation-game trajectories
against the reference. The measured result above is recorded in
`verification-report.json`; reruns should use that report rather than an
assumed tolerance or speed claim in this document.

`cudabench` uses deterministic golden inputs with separate cold and warm
measurements. The standard benchmark run on the selected export recorded a
**421,870 ns** cold call. Across 200 warm calls it recorded **78,260 ns min**,
**94,945 ns mean**, **94,010 ns p50**, **111,400 ns p95**, and **125,670 ns
max**. The JSON report records the model and device identity alongside those
numbers. They describe this batch-one workload only; they are not a comparison
with GoMLX or evidence of a general speedup. Launch overhead and transfers can
matter materially at this size.

The direct demo listens on <http://127.0.0.1:8083>. It warms one opening
position, exposes `GET /healthz`, `GET`/`PUT /api/models`, `GET /api/solutions`,
and `POST /api/games`. `GET /api/solutions` returns the 2,309-word canonical
solution vocabulary, and the UI accepts any one of those words through a plain
text field. The server remains authoritative: malformed or non-canonical words
are rejected. The UI shows `Inference backend: hand-written CUDA via cgo`,
model identity, and GPU identity. It does not hot-swap models or proxy browser
calls to a separate process. The retained GoMLX visualiser remains at port
8082.

This interactive choice deliberately extends beyond the 100 validation words.
`cudaweb` still uses `vocabulary.LoadWithoutFinalTest`, so it never opens the
separate final-test split file or learns which canonical words belong to that
split. A user-directed game may nevertheless name a word from the all-solutions
vocabulary that happens to be held out. Treat those games only as demo output:
the golden-vector parity gate, 100-game trajectory check, and reported metrics
above remain validation-only evidence.

Audit the direct runtime after building it from inside the development
container:

```console
go list -deps -tags cuda_cgo ./cmd/cudaweb
go version -m bin/cudaweb
ldd bin/cudaweb
```

The dependency audit must show no GoMLX, PJRT, or XLA package in `cudaweb` or
`cudaverify`; the offline exporter is intentionally exempt.

## Profiling and screenshot assets

The native wrapper emits NVTX ranges named `wordle_infer`, `copy_inputs_h2d`,
`forward_pass`, and `copy_logits_d2h` on the locked worker thread. The seven
kernel names remain visible in a profile.

```console
make cuda-cgo-profile-systems MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
make cuda-cgo-profile-compute MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

Systems 2025.6.3 writes
[`wordle-inference.nsys-rep`](../../artifacts/cuda-cgo/nsight-systems/wordle-inference.nsys-rep)
and textual summaries under `artifacts/cuda-cgo/nsight-systems/`. It uses a
read-only mount of the locally installed host Nsight directory into the normal
container; it does not install software or change host drivers. A checked
`cudaDeviceReset()` at native teardown gives CUPTI an explicit finalization
point, so the report contains the NVTX ranges, three HtoD copies, seven named
kernels, DtoH copy, and synchronization.

Nsight Compute 2025.4 writes
[`policy-logits.ncu-rep`](../../artifacts/cuda-cgo/nsight-compute/policy-logits.ncu-rep),
a CSV details summary, and source output. For the single
`policy_logits_with_bonus` launch it measured a 4,739 × 128 launch, **11.36 µs**
duration, **69.89% achieved occupancy** versus 100% theoretical, 40
registers/thread, 16 B static plus 1.02 KB driver shared memory per block, no
local or shared-memory spills, 270.81 GB/s memory throughput, and 30.83% DRAM
throughput. Source correlation is working for the FMA and reduction lines.
Those observations identify behavior for this one kernel; occupancy and memory
throughput are diagnostics, not direct performance verdicts.

The exact commands, read-only mount mechanism, and report interpretation are
maintained in [`artifacts/cuda-cgo/README.md`](../../artifacts/cuda-cgo/README.md).

Five slide-capture recipes, including exact real-code ranges, are in
[CUDA/cgo screenshot guide](../../talk/cuda-cgo-screenshot-guide.md). The
checked-in browser evidence is
[`browser-demo.png`](../../talk/assets/cuda-cgo/browser-demo.png), showing
`ADEPT` solved in three guesses on the CUDA/cgo route. Nsight GUI screenshots
remain manual even though their report files now exist.

## Deliberate limitations

- One exported model is loaded per `cudaweb` process; there is no model pool,
  batching, hot swapping, multiple stream, CUDA Graph, or CPU fallback.
- Only batch-one fixed Wordle dimensions, FP32, and `sm_120` are supported.
- CUDA implements only the forward pass. Training, reference generation,
  legality, softmax/loss, and gameplay remain outside it.
- Existing GoMLX serving is retained as an independent route and not silently
  replaced.
- A profiler report and measured benchmark are required before saying this
  route is faster. Occupancy is a diagnostic, not a direct performance result.
