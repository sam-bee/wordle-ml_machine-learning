#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr char kExpectedGPU[] = "NVIDIA GeForce RTX 5070 Ti";

void check(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return;
  }

  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
  std::exit(EXIT_FAILURE);
}

__global__ void add(const float* left, const float* right, float* result) {
  *result = *left + *right;
}

}  // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count != 1) {
    std::fprintf(stderr, "expected exactly one visible GPU, found %d\n", device_count);
    return EXIT_FAILURE;
  }

  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  if (std::strcmp(properties.name, kExpectedGPU) != 0) {
    std::fprintf(stderr, "expected %s, found %s\n", kExpectedGPU, properties.name);
    return EXIT_FAILURE;
  }
  if (properties.major != 12 || properties.minor != 0) {
    std::fprintf(stderr, "expected compute capability 12.0, found %d.%d\n",
                 properties.major, properties.minor);
    return EXIT_FAILURE;
  }

  const float left = 19.0F;
  const float right = 23.0F;
  float* device_left = nullptr;
  float* device_right = nullptr;
  float* device_result = nullptr;
  check(cudaMalloc(&device_left, sizeof(float)), "cudaMalloc(left)");
  check(cudaMalloc(&device_right, sizeof(float)), "cudaMalloc(right)");
  check(cudaMalloc(&device_result, sizeof(float)), "cudaMalloc(result)");
  check(cudaMemcpy(device_left, &left, sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(left)");
  check(cudaMemcpy(device_right, &right, sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(right)");

  add<<<1, 1>>>(device_left, device_right, device_result);
  check(cudaGetLastError(), "add kernel launch");

  float result = 0.0F;
  check(cudaMemcpy(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy(result)");
  check(cudaFree(device_left), "cudaFree(left)");
  check(cudaFree(device_right), "cudaFree(right)");
  check(cudaFree(device_result), "cudaFree(result)");

  if (std::fabs(result - 42.0F) > 1e-6F) {
    std::fprintf(stderr, "unexpected CUDA result: %.1f, want 42.0\n", result);
    return EXIT_FAILURE;
  }

  std::printf("CUDA smoke passed: gpu=%s, compute=%d.%d, result=%.1f\n",
              properties.name, properties.major, properties.minor, result);
  return EXIT_SUCCESS;
}
