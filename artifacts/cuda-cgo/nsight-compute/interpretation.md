# Nsight Compute: policy logits kernel

Generated with:

```bash
make cuda-cgo-profile-compute \
  MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

Nsight Compute 2025.4.0 profiled one
`policy_logits_with_bonus` launch on the approved NVIDIA GeForce RTX 5070 Ti
(compute capability 12.0). The replay duration was 11.36 microseconds. The
launch is the intended 4,739 blocks by 128 threads: four warps per block and
606,592 logical threads, which are scheduled on the GPU in waves rather than
all being CPU-like threads executing at once.

Launch resources were 40 registers per thread, 16 bytes of static shared
memory per block, 1.02 KiB of driver shared memory per block, and no dynamic
shared memory. Both local-memory and shared-memory spill requests were zero.

The profiler reports 100% theoretical occupancy and 69.89% achieved occupancy.
Occupancy is a scheduling observation, not a direct measure of performance or
a speedup claim. The memory-workload section reports 270.81 GB/s throughput,
30.83% of the profiler's DRAM-bandwidth estimate. Those numbers describe the
single collected launch; they do not by themselves establish a bottleneck or a
reason to optimise the teaching-oriented implementation.

Source correlation is present for the real CUDA source:

- line 292: `policy_logits_with_bonus` declaration;
- line 303: output-row FP32 fused multiply-add loop;
- line 305: warp shuffle reduction;
- line 313: final first-warp reduction.

The raw `policy-logits.ncu-rep` is ignored and intentionally uncommitted. The
generated CSV and CUDA/SASS source text are reproducible with the command
above; this Markdown file is the compact, trackable interpretation for the
talk material.
