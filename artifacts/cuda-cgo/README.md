# CUDA/cgo profiler artifacts

The generated Nsight reports are deliberately ignored. This directory records
the exact commands that create them without committing large binary reports or
model weights.

Both profiling targets send the instrumented benchmark's JSON report to
`/tmp` inside their short-lived container. They therefore do not overwrite the
canonical 20-warm-up/200-measurement `benchmark-report.json` beside the model.

Run all commands from the repository root. `MODEL_DIR` may be either an
absolute in-container path or a path relative to the repository root:

```bash
make cuda-cgo-profile-systems \
  MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
make cuda-cgo-profile-compute \
  MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

`cuda-cgo-profile-systems` writes `wordle-inference.nsys-rep` and column-format
summaries for CUDA grid/block launches, CUDA memory operations, and NVTX
push/pop ranges under `nsight-systems/`. It uses:

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
nsys stats --format=column --report=cuda_api_sum,cuda_gpu_kern_gb_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace,nvtx_pushpop_sum
```

The CUDA 13.1 development image provides `nvcc` and `ncu`, but no compatible
Nsight Systems package is available from its configured apt sources:
`nsight-systems-cli` and `nsight-systems` both have no install candidate. The
Systems target therefore mounts the complete version directory containing the
already installed host `nsys` target **read-only** into its normal unprivileged
GPU container. It discovers that directory from:

```bash
readlink -f "$(command -v nsys)"
```

For example, host `nsys` at
`/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nsys` makes the target
mount `/opt/nvidia/nsight-systems/2025.6.3`. The short-lived container creates
only a `/tmp/wordleml-nsys` symlink to the mounted
`target-linux-x64/nsys`: this is required by the Nsight CLI's installation
layout check. No host software, driver setting, or project image is modified.
To use another local installation, set its version root explicitly:

```bash
make cuda-cgo-profile-systems \
  NSYS_HOST_DIR=/opt/nvidia/nsight-systems/2025.6.3 \
  MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

This route has been run successfully with Nsight Systems 2025.6.3 against the
approved RTX 5070 Ti in the CUDA 13.1 container. The real `cudabench` capture
produced the `.nsys-rep`, SQLite export, and CUDA API/kernel/memory/NVTX text
summaries under `nsight-systems/`; see its committed
[`interpretation.md`](nsight-systems/interpretation.md).

The Go runtime does not run the CUDA runtime's usual C exit cleanup. The
native `wordle_cuda_model_destroy` therefore performs a checked
`cudaDeviceReset()` after it has freed the model's persistent buffers and
destroyed its stream. This is teardown only, never inference: it gives
CUPTI/Nsight Systems an explicit finalization point to flush the CUDA API,
kernel, and copy records before the Go process exits. Without it, the report
can contain correct NVTX ranges but empty CUDA kernel and memory tables.

`cuda-cgo-profile-compute` writes the `policy-logits.ncu-rep` report plus its
CSV details and CUDA/SASS source view under `nsight-compute/`. It uses the
locally verified syntax:

```text
ncu --set full -k regex:^policy_logits_with_bonus.* -s 20 -c 1
ncu --page source --print-source cuda,sass
```

The host driver has `RmProfilingAdminOnly: 1`. A normal container user cannot
collect counters (`ERR_NVGPUCTRPERM`); an ephemeral root container with only
`CAP_SYS_ADMIN` can, and has been verified against the existing CUDA smoke
kernel. The Make target grants that capability only to its one NCU container,
then restores ownership of generated report files to the invoking host user.
The regular native build, tests, benchmark, and browser demo stay unprivileged.

Interpret the reports as follows:

- Systems is the timeline: expand the `wordle-gpu` host thread and the
  `wordle_infer` NVTX range, then inspect the three HtoD copies, seven named
  kernels, logits DtoH copy, and final synchronization.
- Compute is the one-kernel view: select `policy_logits_with_bonus`; record its
  4,739-block by 128-thread configuration, occupancy, registers, shared
  memory, memory workload, and the source-correlated FMA/reduction lines.

Profiler output is evidence, not a speedup claim. In particular, occupancy is
not a direct measurement of performance.
