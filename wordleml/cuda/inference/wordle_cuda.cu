// Fixed-shape, FP32-only Wordle policy inference for the conference demo.
//
// The code purposefully favours a direct mapping from the documented network
// to seven profiler-visible kernels over a general tensor framework.  One
// 128-thread block computes one dense output, reducing four warp subtotals in
// shared memory.  The host launches all seven kernels into one stream.

#include "wordle_cuda.h"

#include <cuda_runtime.h>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#define WORDLE_CUDA_HAS_NVTX 1
#include <nvtx3/nvToolsExt.h>
#endif
#endif
#ifndef WORDLE_CUDA_HAS_NVTX
#define WORDLE_CUDA_HAS_NVTX 0
#endif

#include <pthread.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

constexpr int kNumSolutions = WORDLE_CUDA_NUM_SOLUTIONS;
constexpr int kNumActions = WORDLE_CUDA_NUM_ACTIONS;
constexpr int kCandidateStatsSize = WORDLE_CUDA_CANDIDATE_STATS_SIZE;
constexpr int kNumTurns = WORDLE_CUDA_NUM_TURNS;
constexpr int kCandidateProjectionWidth = 96;
constexpr int kStatsProjectionWidth = 48;
constexpr int kTurnEmbeddingWidth = 16;
constexpr int kTrunkSize = WORDLE_CUDA_TRUNK_SIZE;
constexpr int kDenseThreads = 128;
constexpr int kTurnThreads = 32;
constexpr int kWarpsPerDenseBlock = kDenseThreads / 32;

// These are offsets in the one, output-major, FP32 weight allocation. They
// match the exported artifact manifest and are deliberately fixed here rather
// than parsed in CUDA C++.
constexpr size_t kCandidateProjectionWeightOffset = 0;
constexpr size_t kCandidateProjectionBiasOffset =
    kCandidateProjectionWeightOffset + kCandidateProjectionWidth * kNumSolutions;
constexpr size_t kStatsProjectionWeightOffset =
    kCandidateProjectionBiasOffset + kCandidateProjectionWidth;
constexpr size_t kStatsProjectionBiasOffset =
    kStatsProjectionWeightOffset + kStatsProjectionWidth * kCandidateStatsSize;
constexpr size_t kTurnEmbeddingOffset =
    kStatsProjectionBiasOffset + kStatsProjectionWidth;
constexpr size_t kResidualInWeightOffset =
    kTurnEmbeddingOffset + kNumTurns * kTurnEmbeddingWidth;
constexpr size_t kResidualInBiasOffset =
    kResidualInWeightOffset + kTrunkSize * kTrunkSize;
constexpr size_t kResidualOutWeightOffset =
    kResidualInBiasOffset + kTrunkSize;
constexpr size_t kResidualOutBiasOffset =
    kResidualOutWeightOffset + kTrunkSize * kTrunkSize;
constexpr size_t kBaseLogitsWeightOffset =
    kResidualOutBiasOffset + kTrunkSize;
constexpr size_t kBaseLogitsBiasOffset =
    kBaseLogitsWeightOffset + kNumActions * kTrunkSize;
constexpr size_t kCandidateBonusWeightOffset =
    kBaseLogitsBiasOffset + kNumActions;
constexpr size_t kCandidateBonusBiasOffset =
    kCandidateBonusWeightOffset + kTrunkSize;
constexpr size_t kWeightCount = kCandidateBonusBiasOffset + 1;

static_assert(kWeightCount == WORDLE_CUDA_WEIGHT_COUNT,
              "The CUDA layout must contain exactly the documented parameter count");
static_assert(kWarpsPerDenseBlock == 4,
              "The teaching reduction expects four warps per dense block");

constexpr char kRTX5070Ti[] = "NVIDIA GeForce RTX 5070 Ti";
constexpr char kRTX5050[] = "NVIDIA GeForce RTX 5050";
constexpr char kRTX5050Laptop[] = "NVIDIA GeForce RTX 5050 Laptop GPU";

thread_local char g_last_error[512] = "";

void clear_error() { g_last_error[0] = '\0'; }

void set_error(const char* format, ...) {
  va_list args;
  va_start(args, format);
  std::vsnprintf(g_last_error, sizeof(g_last_error), format, args);
  va_end(args);
}

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  set_error("%s failed: %s", operation, cudaGetErrorString(status));
  return false;
}

bool is_approved_gpu(const char* name) {
  return std::strcmp(name, kRTX5070Ti) == 0 ||
         std::strcmp(name, kRTX5050) == 0 ||
         std::strcmp(name, kRTX5050Laptop) == 0;
}

#if WORDLE_CUDA_HAS_NVTX
#define WORDLE_NVTX_PUSH(name) nvtxRangePushA(name)
#define WORDLE_NVTX_POP() nvtxRangePop()
#else
#define WORDLE_NVTX_PUSH(name) ((void)(name))
#define WORDLE_NVTX_POP() ((void)0)
#endif

// Reduces a register subtotal inside a 32-lane warp. Dense kernels below use
// shared memory only for the four warp leaders, never for their input vectors.
__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int delta = 16; delta > 0; delta /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, delta);
  }
  return value;
}

// 96 blocks x 128 threads: one normalized candidate-mask projection output
// per block. Weight rows are output-major, so adjacent threads read adjacent
// FP32 weights in a row.
extern "C" __global__ void candidate_projection_relu(
    const float* candidate_mask, float candidate_reciprocal,
    const float* weights, const float* bias, float* h) {
  __shared__ float warp_sums[kWarpsPerDenseBlock];
  const int output = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sum = 0.0F;
  const float* row = weights + output * kNumSolutions;
  for (int index = threadIdx.x; index < kNumSolutions; index += blockDim.x) {
    // Apply the host-computed reciprocal before the dense product so this is
    // the documented `candidateMask / row sum -> Linear` graph, not merely a
    // mathematically similar post-reduction scaling.
    sum = fmaf(row[index], candidate_mask[index] * candidate_reciprocal, sum);
  }
  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerDenseBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      const float value = block_sum + bias[output];
      h[output] = value > 0.0F ? value : 0.0F;
    }
  }
}

// 48 blocks x 128 threads: the statistics branch fills h[96:144].
extern "C" __global__ void stats_projection_relu(const float* candidate_stats,
                                                   const float* weights,
                                                   const float* bias, float* h) {
  __shared__ float warp_sums[kWarpsPerDenseBlock];
  const int output = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sum = 0.0F;
  const float* row = weights + output * kCandidateStatsSize;
  for (int index = threadIdx.x; index < kCandidateStatsSize; index += blockDim.x) {
    sum = fmaf(row[index], candidate_stats[index], sum);
  }
  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerDenseBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      const float value = block_sum + bias[output];
      h[kCandidateProjectionWidth + output] = value > 0.0F ? value : 0.0F;
    }
  }
}

// 1 block x 32 threads: copy the selected 16-value embedding into h[144:160].
extern "C" __global__ void load_turn_embedding(int32_t turn,
                                                 const float* turn_embedding,
                                                 float* h) {
  const int index = threadIdx.x;
  if (index < kTurnEmbeddingWidth) {
    h[kCandidateProjectionWidth + kStatsProjectionWidth + index] =
        turn_embedding[turn * kTurnEmbeddingWidth + index];
  }
}

// 160 blocks x 128 threads: first residual dense layer, followed by ReLU.
extern "C" __global__ void residual_in_relu(const float* h,
                                              const float* weights,
                                              const float* bias,
                                              float* residual) {
  __shared__ float warp_sums[kWarpsPerDenseBlock];
  const int output = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sum = 0.0F;
  const float* row = weights + output * kTrunkSize;
  for (int index = threadIdx.x; index < kTrunkSize; index += blockDim.x) {
    sum = fmaf(row[index], h[index], sum);
  }
  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerDenseBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      const float value = block_sum + bias[output];
      residual[output] = value > 0.0F ? value : 0.0F;
    }
  }
}

// 160 blocks x 128 threads: second residual layer, skip addition, then ReLU.
extern "C" __global__ void residual_out_skip_relu(const float* residual,
                                                    const float* weights,
                                                    const float* bias,
                                                    float* h) {
  __shared__ float warp_sums[kWarpsPerDenseBlock];
  const int output = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sum = 0.0F;
  const float* row = weights + output * kTrunkSize;
  for (int index = threadIdx.x; index < kTrunkSize; index += blockDim.x) {
    sum = fmaf(row[index], residual[index], sum);
  }
  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerDenseBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      const float value = h[output] + block_sum + bias[output];
      h[output] = value > 0.0F ? value : 0.0F;
    }
  }
}

// 1 block x 128 threads: the learned scalar used only for current candidates.
extern "C" __global__ void candidate_bonus(const float* h,
                                             const float* weights,
                                             const float* bias, float* beta) {
  __shared__ float warp_sums[kWarpsPerDenseBlock];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sum = 0.0F;
  for (int index = threadIdx.x; index < kTrunkSize; index += blockDim.x) {
    sum = fmaf(weights[index], h[index], sum);
  }
  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerDenseBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      beta[0] = block_sum + bias[0];
    }
  }
}

// 4739 blocks x 128 threads: one raw policy logit per action. This keeps the
// learned remaining-candidate bonus distinct from Go's availability mask.
extern "C" __global__ void policy_logits_with_bonus(
    const float* h, const float* remaining_action_mask, const float* weights,
    const float* bias, const float* beta, float* logits) {
  __shared__ float warp_sums[kWarpsPerDenseBlock];
  const int action = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float sum = 0.0F;
  const float* row = weights + action * kTrunkSize;
  for (int index = threadIdx.x; index < kTrunkSize; index += blockDim.x) {
    sum = fmaf(row[index], h[index], sum);
  }
  sum = warp_reduce_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerDenseBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      logits[action] = block_sum + bias[action] +
                       beta[0] * remaining_action_mask[action];
    }
  }
}

}  // namespace

// The opaque model owns every persistent CUDA allocation. It is defined after
// the kernels to make the C ABI in the header remain intentionally compact.
struct wordle_cuda_model {
  int device_ordinal = 0;
  cudaStream_t stream = nullptr;
  float* weights = nullptr;
  float* candidate_mask = nullptr;
  float* candidate_stats = nullptr;
  float* remaining_action_mask = nullptr;
  float* h = nullptr;
  float* residual = nullptr;
  float* beta = nullptr;
  float* logits = nullptr;
  wordle_cuda_model_info info{};
};

namespace {

void capture_cleanup_error(cudaError_t status, const char* operation,
                           char* first_error, size_t first_error_size) {
  if (status == cudaSuccess || first_error[0] != '\0') {
    return;
  }
  std::snprintf(first_error, first_error_size, "%s failed: %s", operation,
                cudaGetErrorString(status));
}

// release_model always attempts every teardown operation. Internal creation
// cleanup passes preserve_existing_error=true so the caller retains the
// operation which actually caused creation to fail; public destruction clears
// stale errors first and reports the first teardown operation which failed.
int release_model(wordle_cuda_model* model, bool preserve_existing_error) {
  if (model == nullptr) {
    return WORDLE_CUDA_OK;
  }

  char existing_error[sizeof(g_last_error)] = "";
  if (preserve_existing_error && g_last_error[0] != '\0') {
    std::snprintf(existing_error, sizeof(existing_error), "%s", g_last_error);
  } else {
    clear_error();
  }

  char first_cleanup_error[sizeof(g_last_error)] = "";
  if (model->device_ordinal >= 0) {
    capture_cleanup_error(cudaSetDevice(model->device_ordinal),
                          "cudaSetDevice(destroy)", first_cleanup_error,
                          sizeof(first_cleanup_error));
  }
  if (model->logits != nullptr) {
    capture_cleanup_error(cudaFree(model->logits), "cudaFree(logits)",
                          first_cleanup_error, sizeof(first_cleanup_error));
  }
  if (model->beta != nullptr) {
    capture_cleanup_error(cudaFree(model->beta), "cudaFree(beta)",
                          first_cleanup_error, sizeof(first_cleanup_error));
  }
  if (model->residual != nullptr) {
    capture_cleanup_error(cudaFree(model->residual), "cudaFree(residual)",
                          first_cleanup_error, sizeof(first_cleanup_error));
  }
  if (model->h != nullptr) {
    capture_cleanup_error(cudaFree(model->h), "cudaFree(h)",
                          first_cleanup_error, sizeof(first_cleanup_error));
  }
  if (model->remaining_action_mask != nullptr) {
    capture_cleanup_error(cudaFree(model->remaining_action_mask),
                          "cudaFree(remaining_action_mask)",
                          first_cleanup_error, sizeof(first_cleanup_error));
  }
  if (model->candidate_stats != nullptr) {
    capture_cleanup_error(cudaFree(model->candidate_stats),
                          "cudaFree(candidate_stats)", first_cleanup_error,
                          sizeof(first_cleanup_error));
  }
  if (model->candidate_mask != nullptr) {
    capture_cleanup_error(cudaFree(model->candidate_mask),
                          "cudaFree(candidate_mask)", first_cleanup_error,
                          sizeof(first_cleanup_error));
  }
  if (model->weights != nullptr) {
    capture_cleanup_error(cudaFree(model->weights), "cudaFree(weights)",
                          first_cleanup_error, sizeof(first_cleanup_error));
  }
  if (model->stream != nullptr) {
    capture_cleanup_error(cudaStreamDestroy(model->stream),
                          "cudaStreamDestroy", first_cleanup_error,
                          sizeof(first_cleanup_error));
  }
  // Go terminates through its own runtime rather than the C library's normal
  // exit path. Releasing the primary context here also gives CUPTI/Nsight an
  // explicit CUDA teardown point at which to flush pending activity records.
  capture_cleanup_error(cudaDeviceReset(), "cudaDeviceReset",
                        first_cleanup_error, sizeof(first_cleanup_error));
  delete model;

  if (existing_error[0] != '\0') {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", existing_error);
    return WORDLE_CUDA_ERROR;
  }
  if (first_cleanup_error[0] != '\0') {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s",
                  first_cleanup_error);
    return WORDLE_CUDA_ERROR;
  }
  return WORDLE_CUDA_OK;
}

int validate_visible_device(wordle_cuda_model_info* info) {
  int device_count = 0;
  if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return WORDLE_CUDA_ERROR;
  }
  if (device_count != 1) {
    set_error("expected exactly one visible CUDA device, found %d", device_count);
    return WORDLE_CUDA_INVALID_DEVICE;
  }
  if (!check_cuda(cudaSetDevice(0), "cudaSetDevice(0)")) {
    return WORDLE_CUDA_ERROR;
  }

  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, 0),
                  "cudaGetDeviceProperties(0)")) {
    return WORDLE_CUDA_ERROR;
  }
  if (!is_approved_gpu(properties.name)) {
    set_error("expected approved RTX 5070 Ti or RTX 5050 (including Laptop GPU), found %s",
              properties.name);
    return WORDLE_CUDA_INVALID_DEVICE;
  }
  if (properties.major != 12 || properties.minor != 0) {
    set_error("expected compute capability 12.0, found %d.%d", properties.major,
              properties.minor);
    return WORDLE_CUDA_INVALID_DEVICE;
  }

  int runtime_version = 0;
  int driver_version = 0;
  if (!check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion") ||
      !check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion")) {
    return WORDLE_CUDA_ERROR;
  }

  std::snprintf(info->device_name, sizeof(info->device_name), "%s", properties.name);
  info->device_ordinal = 0;
  info->compute_major = properties.major;
  info->compute_minor = properties.minor;
  info->cuda_runtime_version = runtime_version;
  info->cuda_driver_version = driver_version;
  info->weight_count = kWeightCount;
  return WORDLE_CUDA_OK;
}

bool allocate_buffer(float** buffer, size_t elements, const char* label) {
  cudaError_t status = cudaMalloc(reinterpret_cast<void**>(buffer), elements * sizeof(float));
  return check_cuda(status, label);
}

bool check_launch(const char* kernel_name) {
  return check_cuda(cudaGetLastError(), kernel_name);
}

// HtoD copies may be asynchronous. Even after a later launch/copy error, do
// not let the C ABI return while CUDA might still dereference a Go-owned host
// pointer supplied for this call. Keep the original failure and append a
// cleanup synchronization failure if the stream also reports one.
int return_after_failed_async_infer(wordle_cuda_model* model) {
  char original_error[sizeof(g_last_error)];
  std::snprintf(original_error, sizeof(original_error), "%s", g_last_error);
  const cudaError_t status = cudaStreamSynchronize(model->stream);
  if (status != cudaSuccess) {
    if (original_error[0] != '\0') {
      set_error("%s; cudaStreamSynchronize(error cleanup) failed: %s",
                original_error, cudaGetErrorString(status));
    } else {
      set_error("cudaStreamSynchronize(error cleanup) failed: %s",
                cudaGetErrorString(status));
    }
  }
  return WORDLE_CUDA_ERROR;
}

}  // namespace

extern "C" int wordle_cuda_model_create(const float* host_weights,
                                         size_t weight_count,
                                         wordle_cuda_model** model_out) {
  clear_error();
  if (model_out == nullptr) {
    set_error("model_out must not be null");
    return WORDLE_CUDA_INVALID_ARGUMENT;
  }
  *model_out = nullptr;
  if (host_weights == nullptr) {
    set_error("host_weights must not be null");
    return WORDLE_CUDA_INVALID_ARGUMENT;
  }
  if (weight_count != kWeightCount) {
    set_error("expected %zu FP32 weights, received %zu", kWeightCount, weight_count);
    return WORDLE_CUDA_INVALID_ARGUMENT;
  }

  wordle_cuda_model_info info{};
  const int device_status = validate_visible_device(&info);
  if (device_status != WORDLE_CUDA_OK) {
    return device_status;
  }

  // The Go worker calls runtime.LockOSThread before creation, so this name is
  // stable and makes the request lane easy to find in an Nsight Systems trace.
  const int thread_name_status = pthread_setname_np(pthread_self(), "wordle-gpu");
  if (thread_name_status != 0) {
    set_error("pthread_setname_np(wordle-gpu) failed: %s",
              std::strerror(thread_name_status));
    return WORDLE_CUDA_ERROR;
  }

  wordle_cuda_model* model = new (std::nothrow) wordle_cuda_model;
  if (model == nullptr) {
    set_error("failed to allocate wordle_cuda_model");
    return WORDLE_CUDA_ERROR;
  }
  model->device_ordinal = info.device_ordinal;
  model->info = info;

  if (!check_cuda(cudaStreamCreateWithFlags(&model->stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags") ||
      !allocate_buffer(&model->weights, kWeightCount, "cudaMalloc(weights)") ||
      !allocate_buffer(&model->candidate_mask, kNumSolutions,
                       "cudaMalloc(candidate_mask)") ||
      !allocate_buffer(&model->candidate_stats, kCandidateStatsSize,
                       "cudaMalloc(candidate_stats)") ||
      !allocate_buffer(&model->remaining_action_mask, kNumActions,
                       "cudaMalloc(remaining_action_mask)") ||
      !allocate_buffer(&model->h, kTrunkSize, "cudaMalloc(h)") ||
      !allocate_buffer(&model->residual, kTrunkSize, "cudaMalloc(residual)") ||
      !allocate_buffer(&model->beta, 1, "cudaMalloc(beta)") ||
      !allocate_buffer(&model->logits, kNumActions, "cudaMalloc(logits)")) {
    (void)release_model(model, true);
    return WORDLE_CUDA_ERROR;
  }

  // Weight memory is initialized exactly once, at model creation. The stream
  // synchronization makes the handle ready before Go begins its warm-up call.
  if (!check_cuda(cudaMemcpyAsync(model->weights, host_weights,
                                  kWeightCount * sizeof(float),
                                  cudaMemcpyHostToDevice, model->stream),
                  "cudaMemcpyAsync(weights HtoD)") ||
      !check_cuda(cudaStreamSynchronize(model->stream),
                  "cudaStreamSynchronize(weights upload)")) {
    (void)release_model(model, true);
    return WORDLE_CUDA_ERROR;
  }

  *model_out = model;
  return WORDLE_CUDA_OK;
}

extern "C" int wordle_cuda_model_infer(
    wordle_cuda_model* model, const float* candidate_mask,
    const float* candidate_stats, int32_t turn,
    const float* remaining_action_mask, float* logits_out) {
  clear_error();
  if (model == nullptr || candidate_mask == nullptr || candidate_stats == nullptr ||
      remaining_action_mask == nullptr || logits_out == nullptr) {
    set_error("model and all input/output pointers must not be null");
    return WORDLE_CUDA_INVALID_ARGUMENT;
  }
  if (turn < 0 || turn >= kNumTurns) {
    set_error("turn must be in [0, %d], received %d", kNumTurns - 1, turn);
    return WORDLE_CUDA_INVALID_ARGUMENT;
  }
  if (!check_cuda(cudaSetDevice(model->device_ordinal), "cudaSetDevice(infer)")) {
    return WORDLE_CUDA_ERROR;
  }

  // Candidate normalization is intentionally host-side: it avoids adding an
  // eighth kernel and rejects the invalid empty candidate set before any copy.
  float candidate_sum = 0.0F;
  for (int index = 0; index < kNumSolutions; ++index) {
    if (!std::isfinite(candidate_mask[index])) {
      set_error("candidate_mask[%d] is not finite", index);
      return WORDLE_CUDA_INVALID_ARGUMENT;
    }
    candidate_sum += candidate_mask[index];
  }
  if (!std::isfinite(candidate_sum) || candidate_sum <= 0.0F) {
    set_error("candidate_mask must have a finite positive sum, received %.9g",
              static_cast<double>(candidate_sum));
    return WORDLE_CUDA_INVALID_ARGUMENT;
  }
  const float candidate_reciprocal = 1.0F / candidate_sum;

  WORDLE_NVTX_PUSH("wordle_infer");
  WORDLE_NVTX_PUSH("copy_inputs_h2d");
  if (!check_cuda(cudaMemcpyAsync(model->candidate_mask, candidate_mask,
                                  kNumSolutions * sizeof(float),
                                  cudaMemcpyHostToDevice, model->stream),
                  "cudaMemcpyAsync(candidate_mask HtoD)") ||
      !check_cuda(cudaMemcpyAsync(model->candidate_stats, candidate_stats,
                                  kCandidateStatsSize * sizeof(float),
                                  cudaMemcpyHostToDevice, model->stream),
                  "cudaMemcpyAsync(candidate_stats HtoD)") ||
      !check_cuda(cudaMemcpyAsync(model->remaining_action_mask,
                                  remaining_action_mask,
                                  kNumActions * sizeof(float),
                                  cudaMemcpyHostToDevice, model->stream),
                  "cudaMemcpyAsync(remaining_action_mask HtoD)")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }
  WORDLE_NVTX_POP();

  WORDLE_NVTX_PUSH("forward_pass");
  candidate_projection_relu<<<kCandidateProjectionWidth, kDenseThreads, 0,
                               model->stream>>>(
      model->candidate_mask, candidate_reciprocal,
      model->weights + kCandidateProjectionWeightOffset,
      model->weights + kCandidateProjectionBiasOffset, model->h);
  if (!check_launch("candidate_projection_relu launch")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }

  stats_projection_relu<<<kStatsProjectionWidth, kDenseThreads, 0,
                           model->stream>>>(
      model->candidate_stats, model->weights + kStatsProjectionWeightOffset,
      model->weights + kStatsProjectionBiasOffset, model->h);
  if (!check_launch("stats_projection_relu launch")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }

  load_turn_embedding<<<1, kTurnThreads, 0, model->stream>>>(
      turn, model->weights + kTurnEmbeddingOffset, model->h);
  if (!check_launch("load_turn_embedding launch")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }

  residual_in_relu<<<kTrunkSize, kDenseThreads, 0, model->stream>>>(
      model->h, model->weights + kResidualInWeightOffset,
      model->weights + kResidualInBiasOffset, model->residual);
  if (!check_launch("residual_in_relu launch")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }

  residual_out_skip_relu<<<kTrunkSize, kDenseThreads, 0, model->stream>>>(
      model->residual, model->weights + kResidualOutWeightOffset,
      model->weights + kResidualOutBiasOffset, model->h);
  if (!check_launch("residual_out_skip_relu launch")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }

  candidate_bonus<<<1, kDenseThreads, 0, model->stream>>>(
      model->h, model->weights + kCandidateBonusWeightOffset,
      model->weights + kCandidateBonusBiasOffset, model->beta);
  if (!check_launch("candidate_bonus launch")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }

  policy_logits_with_bonus<<<kNumActions, kDenseThreads, 0, model->stream>>>(
      model->h, model->remaining_action_mask,
      model->weights + kBaseLogitsWeightOffset,
      model->weights + kBaseLogitsBiasOffset, model->beta, model->logits);
  if (!check_launch("policy_logits_with_bonus launch")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }
  WORDLE_NVTX_POP();

  WORDLE_NVTX_PUSH("copy_logits_d2h");
  if (!check_cuda(cudaMemcpyAsync(logits_out, model->logits,
                                  kNumActions * sizeof(float),
                                  cudaMemcpyDeviceToHost, model->stream),
                  "cudaMemcpyAsync(logits DtoH)") ||
      !check_cuda(cudaStreamSynchronize(model->stream),
                  "cudaStreamSynchronize(infer)")) {
    WORDLE_NVTX_POP();
    WORDLE_NVTX_POP();
    return return_after_failed_async_infer(model);
  }
  WORDLE_NVTX_POP();
  WORDLE_NVTX_POP();
  return WORDLE_CUDA_OK;
}

extern "C" int wordle_cuda_model_get_info(const wordle_cuda_model* model,
                                           wordle_cuda_model_info* info_out) {
  clear_error();
  if (model == nullptr || info_out == nullptr) {
    set_error("model and info_out must not be null");
    return WORDLE_CUDA_INVALID_ARGUMENT;
  }
  *info_out = model->info;
  return WORDLE_CUDA_OK;
}

extern "C" const char* wordle_cuda_last_error(void) {
  return g_last_error[0] == '\0' ? "no error" : g_last_error;
}

extern "C" int wordle_cuda_model_destroy(wordle_cuda_model* model) {
  clear_error();
  return release_model(model, false);
}
