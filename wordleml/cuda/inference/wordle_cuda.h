#ifndef WORDLEML_CUDA_INFERENCE_WORDLE_CUDA_H_
#define WORDLEML_CUDA_INFERENCE_WORDLE_CUDA_H_

// This deliberately small C ABI is the only boundary between the Go control
// plane and the CUDA numerical data plane.  CUDA C++ exceptions never cross
// it: callers receive a non-zero status and can read wordle_cuda_last_error().

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  WORDLE_CUDA_NUM_SOLUTIONS = 2309,
  WORDLE_CUDA_NUM_ACTIONS = 4739,
  WORDLE_CUDA_CANDIDATE_STATS_SIZE = 209,
  WORDLE_CUDA_NUM_TURNS = 6,
  WORDLE_CUDA_TRUNK_SIZE = 160,
  WORDLE_CUDA_WEIGHT_COUNT = 1046596,
  WORDLE_CUDA_DEVICE_NAME_MAX = 256,
};

enum wordle_cuda_status {
  WORDLE_CUDA_OK = 0,
  WORDLE_CUDA_ERROR = 1,
  WORDLE_CUDA_INVALID_ARGUMENT = 2,
  WORDLE_CUDA_INVALID_DEVICE = 3,
};

// The model and all of its device allocations are C-owned.  It is intentionally
// opaque to Go: Go only supplies one complete set of input tensors and gets one
// complete raw-logit vector back for each synchronous call.
typedef struct wordle_cuda_model wordle_cuda_model;

typedef struct wordle_cuda_model_info {
  char device_name[WORDLE_CUDA_DEVICE_NAME_MAX];
  int32_t device_ordinal;
  int32_t compute_major;
  int32_t compute_minor;
  int32_t cuda_runtime_version;
  int32_t cuda_driver_version;
  size_t weight_count;
} wordle_cuda_model_info;

// Creates the persistent CUDA state and copies the one contiguous exported
// weight buffer to device global memory. weight_count must be exactly
// WORDLE_CUDA_WEIGHT_COUNT. All cudaMalloc/cudaFree activity is confined to
// creation/destruction; inference itself performs no allocation.
int wordle_cuda_model_create(const float* host_weights, size_t weight_count,
                             wordle_cuda_model** model_out);

// Runs one complete forward pass synchronously. candidate_mask,
// candidate_stats, remaining_action_mask, and logits_out must respectively
// address 2309, 209, 4739, and 4739 FP32 values. turn is in [0, 5]. The call
// returns only after the DtoH logits copy and stream synchronization complete.
int wordle_cuda_model_infer(wordle_cuda_model* model,
                            const float* candidate_mask,
                            const float* candidate_stats, int32_t turn,
                            const float* remaining_action_mask,
                            float* logits_out);

int wordle_cuda_model_get_info(const wordle_cuda_model* model,
                               wordle_cuda_model_info* info_out);

// The returned text remains valid until the next CUDA ABI call on the same
// host thread. The intended Go worker pins all calls to one OS thread.
const char* wordle_cuda_last_error(void);

// Releases every C-owned CUDA resource. Null destruction is idempotent and
// succeeds. A non-zero result records the first failed teardown operation in
// wordle_cuda_last_error(). Call it on the same locked worker thread that
// created the model.
int wordle_cuda_model_destroy(wordle_cuda_model* model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WORDLEML_CUDA_INFERENCE_WORDLE_CUDA_H_
