#include "cuda_utils.h"
#include <cstdint>
#include <mutex>
#include <unordered_map>
#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#endif

int64_t get_device_attribute(int64_t attribute, int64_t device_id) {
  int device = static_cast<int>(device_id);
  if (device < 0) {
    CUDA_CHECK(cudaGetDevice(&device));
  }

  // Cached per (device, attribute), shared by all threads.
  static std::mutex mutex;
  static std::unordered_map<int64_t, int> cache;
  int64_t const key =
      (static_cast<int64_t>(device) << 32) | static_cast<uint32_t>(attribute);
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  int value;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &value, static_cast<cudaDeviceAttr>(attribute), device));
  cache.emplace(key, value);
  return value;
}

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id) {
  int64_t attribute;
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
  // cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 if not is_hip() else 74

#ifdef USE_ROCM
  attribute = hipDeviceAttributeMaxSharedMemoryPerBlock;
#else
  attribute = cudaDevAttrMaxSharedMemoryPerBlockOptin;
#endif

  return get_device_attribute(attribute, device_id);
}
