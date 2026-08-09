# CUDA/cgo screenshot guide

These are capture recipes for the conference slides. The code crops must come
from the real files below. Generated profiler reports are intentionally ignored
by Git; save small reviewed PNGs, if a graphical session is available, under
`talk/assets/cuda-cgo/` with descriptive names.

The real browser capture
[`browser-demo.png`](assets/cuda-cgo/browser-demo.png) is checked in and shows
`ADEPT` solved in three guesses with the CUDA/cgo backend and GPU identity
visible. Nsight Systems and Nsight Compute GUI capture remains manual: reports
and textual summaries exist, but no claimed UI screenshot has been fabricated.

## 1. Go cgo boundary

**Purpose:** establish that Go makes one visible call for a complete policy
forward pass, rather than hiding CUDA behind an unshown dependency.

**Real-code crop:** use a two-panel 16:9 crop from
[`wordleml/cudainfer/backend_cgo.go`](../wordleml/cudainfer/backend_cgo.go):

- lines **5–10** for `#cgo CFLAGS`, `#cgo LDFLAGS`,
  `#include "wordle_cuda.h"`, and `import "C"`;
- lines **82–97** for the allocation of the output, the one
  `C.wordle_cuda_model_infer(...)` call, and every `runtime.KeepAlive`.

Keep the preamble and call as two small aligned panes rather than shrinking a
larger function until it is illegible. Explain that the native call returns
only after its DtoH copy and stream synchronization, so those pointers are not
retained by C.

## 2. C ABI and CUDA launch

**Purpose:** show the small language boundary and one concrete GPU launch.

**Real-code crop:** use a three-panel crop from:

- [`wordle_cuda.h`](../wordleml/cuda/inference/wordle_cuda.h), lines **11–35**:
  the `extern "C"` guard and opaque `wordle_cuda_model` type;
- [`wordle_cuda.h`](../wordleml/cuda/inference/wordle_cuda.h), lines **47–75**:
  creation, one whole-forward-pass ABI function, information, and destruction;
- [`wordle_cuda.cu`](../wordleml/cuda/inference/wordle_cuda.cu), lines
  **695–698**: the real
  `policy_logits_with_bonus<<<kNumActions, kDenseThreads, 0, model->stream>>>`
  launch.

Point out that `kNumActions` is 4,739 and `kDenseThreads` is 128. The CUDA code
also launches six earlier named kernels; the policy launch is the best slide
example because one block computes one possible-word logit.

## 3. Nsight Systems timeline

**Purpose:** show one complete request end-to-end, including host work and GPU
work.

1. Generate a report with:

   ```console
   make cuda-cgo-profile-systems \
     MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
   ```

2. Open `artifacts/cuda-cgo/nsight-systems/wordle-inference.nsys-rep` in the
   Nsight Systems GUI. The report was captured with host Nsight Systems 2025.6.3
   mounted read-only into the normal container; no host package or driver was
   changed. If a future machine has no `nsys`, use its locally installed version
   root through `NSYS_HOST_DIR` rather than copying a binary into the image.
3. Expand the `wordle-gpu` host thread, its NVTX row, CUDA API row, CUDA memory
   row, and the active GPU’s kernel row. Hide unrelated process, storage,
   scheduler, and idle rows.
4. Zoom into one post-warm-up `wordle_infer` range. It must visibly contain
   `copy_inputs_h2d`, the three HtoD copies, `forward_pass`, all seven named
   kernels, `copy_logits_d2h`, the DtoH copy, and the final synchronization.
5. Crop just those rows plus enough timestamps to show their ordering. Label it
   as a timeline, not a speed comparison.

The companion textual summaries are written beside the report and verify all
seven named kernels. See
[`artifacts/cuda-cgo/README.md`](../artifacts/cuda-cgo/README.md) for the exact
CLI flags and read-only mount setup.

## 4. Nsight Compute policy-kernel details and source

**Purpose:** move from the timeline to one kernel without implying that one
metric explains performance.

1. Generate a report with:

   ```console
   make cuda-cgo-profile-compute \
     MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
   ```

2. Open `artifacts/cuda-cgo/nsight-compute/policy-logits.ncu-rep` and select
   `policy_logits_with_bonus`.
3. Show **Launch Statistics** (4,739 blocks and 128 threads), **Occupancy**,
   **Memory Workload Analysis**, and the source-correlated CUDA/SASS view.
4. In the source view, include the multiply-accumulate loop and warp/shared
   reduction in
   [`wordle_cuda.cu`](../wordleml/cuda/inference/wordle_cuda.cu), lines
   **292–322**. `-lineinfo` in the native build is what makes this correlation
   possible.
5. State the actual report values: 11.36 µs duration, 69.89% achieved versus
   100% theoretical occupancy, 40 registers/thread, 16 B static plus 1.02 KB
   driver shared memory per block, no local/shared spills, 270.81 GB/s memory
   throughput, and 30.83% DRAM throughput. Shared memory is per block, and
   occupancy is a diagnostic rather than a direct performance measurement.

The Make target also writes `policy-logits-summary.csv` and
`policy-logits-source.txt` alongside the `.ncu-rep`; they support a manual GUI
capture but are not a substitute for one on a slide claiming a GUI view.

## 5. Browser demo proving the backend

**Purpose:** start the story with a playable result and visibly identify its
implementation.

The checked-in capture is
[`talk/assets/cuda-cgo/browser-demo.png`](assets/cuda-cgo/browser-demo.png).
It is a 1905 × 1267 PNG from the direct route and shows `ADEPT` solved in three
guesses. To refresh it:

1. Start the direct demo:

   ```console
   make cuda-cgo-demo \
     MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
   ```

2. Open <http://127.0.0.1:8083>, type `ADEPT` (or another canonical solution)
   into the free-text field, and run a complete game. The direct process does
   not load final-test split membership; do not present an interactive game as
   validation or final-test evidence.
3. Wait until the page displays the completed board and keep these real UI
   fields in the crop: `Inference backend: hand-written CUDA via cgo`, the
   `cuda-cgo` status, model run/checkpoint/update, and GPU/device identity.
4. Replace `talk/assets/cuda-cgo/browser-demo.png` only after checking that the
   visible backend/device information belongs to the same running demo and the
   completed trajectory is for the canonical word typed into the field.

The direct demo at port 8083 is one Go process. Do not use a screenshot from
the retained port-8082 GoMLX/XLA proxy visualiser as evidence of the cgo route.
