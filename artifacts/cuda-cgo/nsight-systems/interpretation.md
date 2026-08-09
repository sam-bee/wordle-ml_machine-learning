# Nsight Systems: CUDA/cgo inference capture

Generated with:

```bash
make cuda-cgo-profile-systems \
  MODEL_DIR=runs/seed-replication-20260809-132505Z/exports/cuda-f32-v1/best
```

The capture used Nsight Systems 2025.6.3.541 in a read-only host mount and the
project CUDA 13.1 container. Only the approved NVIDIA GeForce RTX 5070 Ti
(compute capability 12.0) was visible in the container.

The capture contains 41 complete inferences: one cold call, 20 warm-ups and
20 measured calls. The CUDA kernel summary records every required named
kernel 41 times, with the fixed teaching launch geometries:

| Kernel | Grid | Block |
| --- | ---: | ---: |
| `candidate_projection_relu` | 96 | 128 |
| `stats_projection_relu` | 48 | 128 |
| `load_turn_embedding` | 1 | 32 |
| `residual_in_relu` | 160 | 128 |
| `residual_out_skip_relu` | 160 | 128 |
| `candidate_bonus` | 1 | 128 |
| `policy_logits_with_bonus` | 4,739 | 128 |

`policy_logits_with_bonus` accounts for 238,052 ns of device time across its
41 launches (5,806.1 ns average). The other six kernels appear in the same
sequence for every inference. These measurements describe this workload and
are not a comparison with another backend.

The GPU-memory summary has 124 Host-to-Device copies and 41 Device-to-Host
copies. The extra HtoD copy is the one-time weight upload; each inference has
the expected three input HtoD copies and one logits DtoH copy. `cuda_api_sum`
records the corresponding `cudaMemcpyAsync`, `cudaLaunchKernel`,
`cudaStreamSynchronize`, allocation, teardown and `cudaDeviceReset` calls.

In the graphical timeline, expand the `wordle-gpu` thread and select one
`wordle_infer` NVTX range. Its nested `copy_inputs_h2d`, `forward_pass`, and
`copy_logits_d2h` ranges align with the three input copies, seven named
kernels, output copy, and final stream synchronization. The full trace is in
`wordle-inference.nsys-rep`; the generated text summaries are adjacent to this
file.

`cudaDeviceReset()` runs only during native-model destruction, after the
stream and persistent allocations are released. It is necessary to flush CUDA
activity records when a Go/cgo program exits; no reset or allocation occurs in
the inference call. This capture documents request structure, not a speedup
claim.

The binary `wordle-inference.nsys-rep` and its SQLite export are ignored and
intentionally uncommitted. This Markdown interpretation is the compact,
trackable record; the adjacent generated text summaries remain available for
inspection or deliberate commits when useful.
