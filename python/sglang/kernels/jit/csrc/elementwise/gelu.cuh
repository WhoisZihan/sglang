#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sglang {

/// \brief Evaluate NewGELU or QuickGELU in FP32.
template <bool kQuick>
SGL_DEVICE float gelu_f32(float x) {
  if constexpr (kQuick) {
    return x / (1.0f + expf(-1.702f * x));
  } else {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
}

/// \brief Vectorized GELU with a scalar tail; kVecSize=1 handles unaligned pointers.
template <typename T, bool kQuick, bool kUsePDL, uint32_t kVecSize, bool kPersistent>
__global__ void gelu_kernel(T* __restrict__ out, const T* __restrict__ input, uint32_t n) {
  using vec_t = device::AlignedVector<T, kVecSize>;
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t first = tid * kVecSize;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x * kVecSize;
  if (first >= n) return;
  device::PDLWaitPrimary<kUsePDL>();
  for (uint64_t offset = first; offset < n; offset += stride) {
    if (n - offset >= kVecSize) {
      auto values = device::load_as<vec_t>(input, offset / kVecSize);
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        values[i] = device::cast<T>(gelu_f32<kQuick>(device::cast<fp32_t>(values[i])));
      }
      device::store_as<vec_t>(out, values, offset / kVecSize);
    } else {
      for (uint64_t i = offset; i < n; ++i) {
        out[i] = device::cast<T>(gelu_f32<kQuick>(device::cast<fp32_t>(input[i])));
      }
    }
    if constexpr (!kPersistent) break;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

/// \brief Validate and launch GELU at a fixed vector width.
template <typename T, bool kQuick, bool kUsePDL, uint32_t kWidth, bool kPersistent = false>
void launch_gelu(tvm::ffi::TensorView input, tvm::ffi::TensorView out) {
  using namespace host;
  auto N = SymbolicSize{"num_elements"};
  auto device_ = SymbolicDevice{};
  // TensorMatcher stores a span into the temporary shape list. Keep every
  // verification in this full expression so that the list stays alive.
  TensorMatcher({N})
      .template with_dtype<T>()
      .template with_device<kDLCUDA>(device_)
      .ensure_alignment(kWidth * sizeof(T))
      .verify(input)
      .verify(out);
  const auto n = N.unwrap();
  CHECK_HOST(n <= std::numeric_limits<uint32_t>::max()) << "too many GELU elements for 32-bit indexing";
  if (n == 0) return;

  constexpr uint32_t kBlockSize = 256;
  auto blocks = div_ceil(div_ceil(n, static_cast<int64_t>(kWidth)), static_cast<int64_t>(kBlockSize));
  const auto kernel = gelu_kernel<T, kQuick, kUsePDL, kWidth, kPersistent>;
  if constexpr (kPersistent) {
    // Cache occupancy per device and specialization, including during graph capture.
    static runtime::DeviceCacheMap<uint32_t, 0> occupancy_cache;
    const auto device_id = device_.unwrap().device_id;
    const auto occupancy = occupancy_cache.get_cached(device_id, true, [&](int32_t) {
#ifndef USE_ROCM
      tvm::ffi::CUDADeviceGuard guard(device_id);
#endif
      return runtime::get_blocks_per_sm(kernel, kBlockSize);
    });
    CHECK_HOST(occupancy > 0) << "GELU kernel has zero occupancy";
    const auto capacity = static_cast<int64_t>(runtime::get_sm_count(device_id)) * occupancy;
    blocks = std::min(blocks, capacity);
  }
  LaunchKernel(blocks, kBlockSize, device_.unwrap())
      .enable_pdl(kUsePDL)(
          kernel, static_cast<T*>(out.data_ptr()), static_cast<const T*>(input.data_ptr()), static_cast<uint32_t>(n));
}

/// \brief Select access width and use an occupancy-bounded grid for large inputs.
template <typename T, bool kQuick, bool kUsePDL, bool kScalarSmall = true, int kGridMode = 0>
void gelu(tvm::ffi::TensorView input, tvm::ffi::TensorView out) {
  static_assert(kGridMode >= 0 && kGridMode <= 2);
  constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(T);
  const auto aligned = (reinterpret_cast<uintptr_t>(input.data_ptr()) % device::kMaxVecBytes == 0) &&
                       (reinterpret_cast<uintptr_t>(out.data_ptr()) % device::kMaxVecBytes == 0);
  // Scalar accesses provide more parallel work for small/medium inputs.
  // Benchmarks favor scalar through 64K elements and vectors at 1M elements.
  // Keep the vectorized path for larger inputs to reduce memory instructions.
  if (aligned && (!kScalarSmall || input.numel() > 65536)) {
    // 0: automatic; 1: ordinary grid; 2: bounded grid (benchmark override).
    if (kGridMode == 2 || (kGridMode == 0 && input.numel() >= (1LL << 24))) {
      launch_gelu<T, kQuick, kUsePDL, kVecSize, true>(input, out);
    } else {
      launch_gelu<T, kQuick, kUsePDL, kVecSize>(input, out);
    }
  } else {
    if (kGridMode == 2 || (kGridMode == 0 && input.numel() >= (1LL << 24))) {
      launch_gelu<T, kQuick, kUsePDL, 1, true>(input, out);
    } else {
      launch_gelu<T, kQuick, kUsePDL, 1>(input, out);
    }
  }
}

}  // namespace sglang
