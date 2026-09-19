/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Toggle between the original single-element-per-iteration LARGE_M_TOPK
// loop (0) and the 2-elements-per-iteration, latency-hiding version (1).
// Flip this to A/B test; both implementations are kept side by side below.
#define NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION 2
#if NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION > 0
  #define NVFP4_ENABLE_ELTS16
#endif

#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/dispatch_utils.h"
#include "../../cuda_vec_utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>

#include "cuda_utils.h"
#include "nvfp4_utils.cuh"
#include "libtorch_stable/launch_bounds_utils.h"
#include "../../../cuda_compat.h"

#define CVT_FP4_MAX_THREADS_PER_BLOCK (512)
#define CVT_FP4_MIN_THREADS_PER_BLOCK (512)
// Expert boundaries stepped over before the lookup switches to a bisection.
// The common case resolves within the first two.
#define CVT_FP4_EXPERT_PROBE_STEPS (2)

namespace vllm {

// Load one packed vector from global memory, using the 256b streaming load
// where it is available.
template <class PackedVecT>
__device__ __forceinline__ PackedVecT LoadPackedVec(PackedVecT const* ptr) {
#if VLLM_256B_PTX_ENABLED
  static_assert(sizeof(PackedVecT) == sizeof(u32x8_t));
  PackedVecT vec;
  ld256(vec, ptr);
  return vec;
#else
  return *ptr;
#endif
}

// NVFP4 quantization kernel for LARGE_M_TOPK = true (large m_topk optimized
// version). When FUSE_SILU_MUL=true, expects input with gate||up layout and
// fuses SiLU(gate)*up before quantization.
#if NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION == 2
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(
    CVT_FP4_MAX_THREADS_PER_BLOCK,
    VLLM_BLOCKS_PER_SM(CVT_FP4_MAX_THREADS_PER_BLOCK))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols,
                    Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout,
                    uint32_t const* __restrict__ input_offset_by_experts,
                    uint32_t const* __restrict__ output_scale_offset_by_experts,
                    int n_experts) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");
  static_assert(
      CVT_FP4_NUM_THREADS_PER_SF == 1,
      "CVT_FP4_NUM_THREADS_PER_SF should be 1 with NVFP4_ENABLE_ELTS16");

  // Hand out the vecs in whole block-wide chunks rather than in rows: a row is
  // only colsPerRow vecs, so a row-granular split leaves the remainder blocks
  // idle, while a chunk-granular one costs at most one iteration per block and
  // leaves the single partial chunk at the end of the tensor.
  //
  // The chunks are taken grid-strided rather than as one contiguous range per
  // block: block b still gets floor/ceil(chunks / gridDim) of them, so the
  // balance is identical, but the whole grid then advances through the tensor
  // together and DRAM sees one dense sequential window instead of gridDim
  // separate cursors megabytes apart.
  int const colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int const vec_stride = (int)blockDim.x;
  int const total_vecs = numRows * colsPerRow;
  int const block_id = (int)blockIdx.x;
  int const vec_start = block_id * vec_stride + (int)threadIdx.x;
  int const grid_stride = (int)gridDim.x * vec_stride;
  // Lowest row this block touches; blocks may start part way into a row.
  int const row_begin = (block_id * vec_stride) / colsPerRow;

  // Three arrays of n_experts+1 entries each. The stride is rounded up to 4
  // entries (16B) because the vectorized branch below stores each array with
  // int4/float4: a bare n_experts+1 stride both overlaps the neighbouring
  // array and misaligns those stores.
  extern __shared__ __align__(16) uint32_t shared_memory[];
  uint32_t* shared_input_offsets = shared_memory;
  // Per-warp count of the experts starting at or before row_begin. warpSize is
  // not a constant expression, so the bound spells out the warp width.
  __shared__ uint32_t warp_expert_counts[CVT_FP4_MAX_THREADS_PER_BLOCK / 32];
  // warpSize is a power of two, so the warp index and the lane test below are a
  // shift and a mask.
  int const warp_mask = warpSize - 1;
  int const warp_shift = __ffs(warpSize) - 1;

  // Load input offsets into shared memory.
  // If n_experts is larger than 4, use vectorized int4 to save instructions.
  // If n_experts is smaller than 4, read directly.
  // Every thread already holds the offsets it loaded, so it also counts the
  // experts starting at or before row_begin here; summing those counts over the
  // block locates the expert owning row_begin without a second pass over the
  // table.
  {
    int const shared_stride = round_up(n_experts + 1, 4);
    uint32_t* shared_output_scale_offsets = shared_memory + shared_stride;
    float* shared_SFScale =
        reinterpret_cast<float*>(shared_memory + shared_stride * 2);

    uint32_t const probe = row_begin;
    int expert_count = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
        uint32_t const input_offset = input_offset_by_experts[i];
        expert_count += (input_offset <= probe);
        shared_SFScale[i] = SFScale ? SFScale[i] : 1.0f;
        shared_output_scale_offsets[i] = output_scale_offset_by_experts[i];
        shared_input_offsets[i] = input_offset;
      }
      shared_output_scale_offsets[n_experts] =
          output_scale_offset_by_experts[n_experts];
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    } else {
      int n_experts_vec4 = n_experts & ~3;
      const float4 scale4 = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
      for (int i = threadIdx.x * 4; i < n_experts_vec4; i += blockDim.x * 4) {
        int4 const input_offsets4 =
            *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
        expert_count += ((uint32_t)input_offsets4.x <= probe) +
                        ((uint32_t)input_offsets4.y <= probe) +
                        ((uint32_t)input_offsets4.z <= probe) +
                        ((uint32_t)input_offsets4.w <= probe);
        *reinterpret_cast<float4*>(&shared_SFScale[i]) =
            SFScale ? *reinterpret_cast<const float4*>(&SFScale[i]) : scale4;
        *reinterpret_cast<int4*>(&shared_output_scale_offsets[i]) =
            *reinterpret_cast<const int4*>(&output_scale_offset_by_experts[i]);
        *reinterpret_cast<int4*>(&shared_input_offsets[i]) = input_offsets4;
      }
      int i = n_experts_vec4 + threadIdx.x;
      if (i <= n_experts) {
        if (i < n_experts) {
          // SFScale has exactly n_experts entries -- no [n_experts] sentinel.
          shared_SFScale[i] = SFScale ? SFScale[i] : 1.0f;
        }
        uint32_t const input_offset = input_offset_by_experts[i];
        expert_count += (i < n_experts) && (input_offset <= probe);
        shared_output_scale_offsets[i] = output_scale_offset_by_experts[i];
        shared_input_offsets[i] = input_offset;
      }
    }

    // Publish one partial count per warp; the block-wide sum is folded into
    // the barrier the offset table already needs.
    expert_count = __reduce_add_sync(0xffffffffu, expert_count);
    if ((threadIdx.x & warp_mask) == 0) {
      warp_expert_counts[threadIdx.x >> warp_shift] = expert_count;
    }
  }
  __syncthreads();

  // Expert holding row_begin, the lowest row this block touches. Rows only ever
  // advance from there, so the loop walks the expert index forward instead of
  // searching again.
  int expert_idx = -1;
  {
    int const num_warps = (int)(blockDim.x >> warp_shift);
    for (int w = 0; w < num_warps; ++w) {
      expert_idx += warp_expert_counts[w];
    }
    expert_idx = max(expert_idx, 0);
  }

  // Flat vec index into the whole tensor.
  for (int vecIdx = vec_start; vecIdx < total_vecs; vecIdx += grid_stride) {
    int const rowIdx = vecIdx / colsPerRow;
    int const colIdx = vecIdx % colsPerRow;

    // Rows only advance, so the expert is at or after the last one. Step over
    // the first few boundaries -- crossing none or one is the usual case --
    // then bisect, so that a lane which jumps over many experts at once still
    // costs log2(n_experts) dependent shared loads rather than one per expert.
    int probes = 0;
    while (expert_idx + 1 < n_experts && probes < CVT_FP4_EXPERT_PROBE_STEPS &&
           shared_input_offsets[expert_idx + 1] <= (uint32_t)rowIdx) {
      ++expert_idx;
      ++probes;
    }
    if (probes == CVT_FP4_EXPERT_PROBE_STEPS) {
      // Last expert whose first row is at or before this one.
      int lo = expert_idx, hi = n_experts - 1;
      while (lo < hi) {
        int const mid = (lo + hi + 1) >> 1;
        bool const le = shared_input_offsets[mid] <= (uint32_t)rowIdx;
        lo = le ? mid : lo;
        hi = le ? hi : mid - 1;
      }
      expert_idx = lo;
    }

    // recompute these every loop to save on registers
    int const shared_stride = round_up(n_experts + 1, 4);
    uint32_t* shared_output_scale_offsets = shared_memory + shared_stride;
    float* shared_SFScale =
        reinterpret_cast<float*>(shared_memory + shared_stride * 2);
    int32_t const numKTiles = (numCols + 63) / 64;

    PackedVec in_vec;

    // read
    {
      // When fusing SiLU+Mul, input has gate || up layout (doubled width)
      int const inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

      PackedVec const* in_row = reinterpret_cast<PackedVec const*>(in) +
                                (int64_t)rowIdx * inColsPerRow;

      in_vec = LoadPackedVec(in_row + colIdx);

      if constexpr (FUSE_SILU_MUL) {
        PackedVec const in_vec_up = LoadPackedVec(in_row + colsPerRow + colIdx);
        in_vec = compute_silu_mul(in_vec, in_vec_up);
      }
    }

    // write
    {
      uint64_t* out_row =
          reinterpret_cast<uint64_t*>(out) +
          (int64_t)rowIdx * (colsPerRow / CVT_FP4_NUM_THREADS_PER_SF);

      uint8_t* const sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx - shared_input_offsets[expert_idx], colIdx, numKTiles,
              SFout + shared_output_scale_offsets[expert_idx] * numKTiles);

      uint8_t sf_val;
      u32x2 const o =
          cvt_warp_fp16_to_fp4_and_sf<Type, CVT_FP4_NUM_THREADS_PER_SF,
                                      UE8M0_SF>(
              in_vec, shared_SFScale[expert_idx], sf_val);
      *sf_out = sf_val;

      st64_cs(reinterpret_cast<int64_t*>(out_row + colIdx),
              static_cast<int64_t>(static_cast<uint64_t>(o.hi) << 32 | o.lo));
    }
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  if (m_topk == 0) return;

  // NOTE: get_device_attribute() caches the first attribute it is ever asked
  // for and returns it for every later call, so query only one attribute here.
  int const multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  // Block size. The kernel hands out work in chunks of one vec per thread, so
  // the block size is both the balance granularity and the occupancy knob: aim
  // for a single chunk per resident block slot, so that every SM gets work and
  // no SM gets a whole chunk more than its neighbours.
  int const warpSize = WARP_SIZE;
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int64_t const total_thread_work = (int64_t)m_topk * workSizePerRow;
  int64_t const block_slots =
      (int64_t)multiProcessorCount * VLLM_LAUNCH_BLOCKS_CAP;
  int64_t const threads_per_block =
      round_up(div_round_up(total_thread_work, block_slots), (int64_t)warpSize);
  // The upper bound is the kernel's __launch_bounds__ -- a larger block fails
  // the launch with cudaErrorInvalidValue. The lower bound keeps enough warps
  // in flight to hide the load latency and amortizes the expert table load,
  // which every block repeats.
  dim3 block(std::clamp<int64_t>(threads_per_block,
                                 CVT_FP4_MIN_THREADS_PER_BLOCK,
                                 CVT_FP4_MAX_THREADS_PER_BLOCK));

  // Exactly one wave: fill every resident block slot on the device once and
  // let each block iterate over its own contiguous range of vecs. The kernel
  // hands out whole block-wide chunks, so cap the grid at the chunk count
  // rather than launching blocks with nothing to do.
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  int64_t const total_chunks =
      div_round_up(total_thread_work, (int64_t)block.x);
  dim3 grid(std::min<int64_t>(total_chunks,
                              (int64_t)multiProcessorCount * numBlocksPerSM));

  // shared memory for the expert offset table: three arrays of n_experts+1
  // entries, each padded to a 16B-aligned stride.
  size_t const shared_mem_size =
      3 * round_up(n_experts + 1, 4) * sizeof(uint32_t);

  // Use the scalar expert load if n_experts >= 4, otherwise
  // use vectorized load
  if (n_experts < 4) {
    cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, true>
        <<<grid, block, shared_mem_size, stream>>>(
            m_topk, k, reinterpret_cast<T*>(input),
            reinterpret_cast<float*>(input_global_scale),
            reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(output_scale),
            reinterpret_cast<uint32_t*>(input_offset_by_experts),
            reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
            n_experts);
  } else {
    cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, false>
        <<<grid, block, shared_mem_size, stream>>>(
            m_topk, k, reinterpret_cast<T*>(input),
            reinterpret_cast<float*>(input_global_scale),
            reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(output_scale),
            reinterpret_cast<uint32_t*>(input_offset_by_experts),
            reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
            n_experts);
  }

  STD_CUDA_KERNEL_LAUNCH_CHECK();
}
#elif NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION == 1
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(
    CVT_FP4_MAX_THREADS_PER_BLOCK,
    VLLM_BLOCKS_PER_SM(CVT_FP4_MAX_THREADS_PER_BLOCK))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols,
                    Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout,
                    uint32_t* __restrict__ input_offset_by_experts,
                    uint32_t* __restrict__ output_scale_offset_by_experts,
                    int n_experts) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int const colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int const total_vecs = numRows * colsPerRow;
  int const vec_start = blockIdx.x * blockDim.x + threadIdx.x;
  int const row_start = (blockIdx.x * blockDim.x) / colsPerRow;

  extern __shared__ uint32_t shared_input_offsets[];

  // Load input offsets into shared memory.
  // If n_experts is larger than 4, use vectorized int4 to save instructions.
  // If n_experts is smaller than 4, read directly.
  if constexpr (SMALL_NUM_EXPERTS) {
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) =
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t const numValidRows =
      min(numRows, static_cast<int32_t>(shared_input_offsets[n_experts]));
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numValidRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find expert using binary search for better performance with large m_topk
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // Binary search through experts using shared memory
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      // Get offsets: shared_input_offsets[i] corresponds to
      // input_offset_by_experts[i]
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // Load input and optionally apply fused SiLU+Mul
    int64_t inOffset = rowIdx * inColsPerRow + colIdx;
    PackedVec in_vec =
        LoadPackedVec(reinterpret_cast<PackedVec const*>(in) + inOffset);
    PackedVec quant_input;
    if constexpr (FUSE_SILU_MUL) {
      PackedVec in_vec_up = LoadPackedVec(
          reinterpret_cast<PackedVec const*>(in) + colsPerRow + inOffset);
      quant_input = compute_silu_mul(in_vec, in_vec_up);
    } else {
      quant_input = in_vec;
    }

    uint64_t* out_row =
        reinterpret_cast<uint64_t*>(out) +
        (int64_t)rowIdx * (colsPerRow / CVT_FP4_NUM_THREADS_PER_SF);

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    u32x2 const o =
        cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
            quant_input, SFScaleVal, sf_out);

    st64_cs(reinterpret_cast<int64_t*>(out_row + colIdx),
            static_cast<int64_t>(static_cast<uint64_t>(o.hi) << 32 | o.lo));
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  // Block size. The kernel hands out work in chunks of one vec per thread, so
  // the block size is both the balance granularity and the occupancy knob: aim
  // for a single chunk per resident block slot, so that every SM gets work and
  // no SM gets a whole chunk more than its neighbours.
  int const warpSize = WARP_SIZE;
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int64_t const total_thread_work = (int64_t)m_topk * workSizePerRow;
  int64_t const block_slots =
      (int64_t)multiProcessorCount * VLLM_LAUNCH_BLOCKS_CAP;
  int64_t const threads_per_block =
      round_up(div_round_up(total_thread_work, block_slots), (int64_t)warpSize);
  // The upper bound is the kernel's __launch_bounds__ -- a larger block fails
  // the launch with cudaErrorInvalidValue. The lower bound keeps enough warps
  // in flight to hide the load latency and amortizes the expert table load,
  // which every block repeats.
  dim3 block(std::clamp<int64_t>(threads_per_block,
                                 CVT_FP4_MIN_THREADS_PER_BLOCK,
                                 CVT_FP4_MAX_THREADS_PER_BLOCK));

  // Exactly one wave: fill every resident block slot on the device once and
  // let each block iterate over its own contiguous range of vecs. The kernel
  // hands out whole block-wide chunks, so cap the grid at the chunk count
  // rather than launching blocks with nothing to do.
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  int64_t const total_chunks =
      div_round_up(total_thread_work, (int64_t)block.x);
  dim3 grid(std::min<int64_t>(total_chunks,
                              (int64_t)multiProcessorCount * numBlocksPerSM));

  size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
  // The shared-memory vectorized offset load only handles full 4-expert
  // chunks. Use the scalar specialization for the remainder cases.
  if (n_experts >= 4 && n_experts % 4 == 0) {
    cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, false>
        <<<grid, block, shared_mem_size, stream>>>(
            m_topk, k, reinterpret_cast<T*>(input),
            reinterpret_cast<float*>(input_global_scale),
            reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(output_scale),
            reinterpret_cast<uint32_t*>(input_offset_by_experts),
            reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
            n_experts);
  } else {
    cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, true>
        <<<grid, block, shared_mem_size, stream>>>(
            m_topk, k, reinterpret_cast<T*>(input),
            reinterpret_cast<float*>(input_global_scale),
            reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(output_scale),
            reinterpret_cast<uint32_t*>(input_offset_by_experts),
            reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
            n_experts);
  }
}
#else  // NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION

// NVFP4 quantization kernel for experts (low-latency path).
// When FUSE_SILU_MUL=true, expects input with gate||up layout and fuses
// SiLU(gate)*up before quantization.
// Use UE4M3 by default.
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout,
                    uint32_t* input_offset_by_experts,
                    uint32_t* output_scale_offset_by_experts, int n_experts,
                    bool low_latency) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int32_t const numValidRows =
      min(numRows,
          static_cast<int32_t>(__ldca(&input_offset_by_experts[n_experts])));
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numValidRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find index within the experts using different strategies based on expert
    // count
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      for (int i = 0; i < n_experts; i++) {
        uint32_t current_offset = __ldca(&input_offset_by_experts[i]);
        uint32_t next_offset = __ldca(&input_offset_by_experts[i + 1]);
        if (rowIdx >= current_offset && rowIdx < next_offset) {
          rowIdx_in_expert = rowIdx - current_offset;
          expert_idx = i;
          break;
        }
      }
    } else {
      // Load input offsets into registers first, then do the computation.
      // Local array size set to 17 because of register limit.
      uint32_t local_offsets[17];
      for (int chunk_start = 0; chunk_start < n_experts; chunk_start += 16) {
        *reinterpret_cast<int4*>(local_offsets) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start]));
        *reinterpret_cast<int4*>(local_offsets + 4) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 4]));
        *reinterpret_cast<int4*>(local_offsets + 8) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 8]));
        *reinterpret_cast<int4*>(local_offsets + 12) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 12]));
        local_offsets[16] = __ldca(&input_offset_by_experts[chunk_start + 16]);

  // Check against the 16 loaded offsets
  #pragma unroll
        for (int i = 0; i < 16; i++) {
          if (rowIdx >= local_offsets[i] && rowIdx < local_offsets[i + 1]) {
            rowIdx_in_expert = rowIdx - local_offsets[i];
            expert_idx = chunk_start + i;
            break;
          }
        }
      }
    }

    // Load input and optionally apply fused SiLU+Mul
    int64_t inOffset = rowIdx * inColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    PackedVec quant_input;
    if constexpr (FUSE_SILU_MUL) {
      PackedVec in_vec_up =
          reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      quant_input = compute_silu_mul(in_vec, in_vec_up);
    } else {
      quant_input = in_vec;
    }

    // Get the output tensor offset.
    // Same as inOffset because 8 elements are packed into one uint32_t.
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // Get the global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is
    // (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
        quant_input, SFScaleVal, sf_out);
  }
}

// Each global thread processes one element: load, then consume.
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false,
          bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(1024, VLLM_BLOCKS_PER_SM(1024))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout,
                    uint32_t* input_offset_by_experts,
                    uint32_t* output_scale_offset_by_experts, int n_experts) {
  using PackedVec = PackedVec<Type, CVT_FP4_PACK16>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  extern __shared__ uint32_t shared_input_offsets[];

  // Load input offsets into shared memory.
  // If n_experts is larger than 4, use vectorized int4 to save instructions.
  // If n_experts is smaller than 4, read directly.
  if constexpr (SMALL_NUM_EXPERTS) {
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) =
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int32_t const numValidRows =
      min(numRows, static_cast<int32_t>(shared_input_offsets[n_experts]));
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numValidRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find expert using binary search for better performance with large m_topk
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // Binary search through experts using shared memory
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      // Get offsets: shared_input_offsets[i] corresponds to
      // input_offset_by_experts[i]
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // Load input and optionally apply fused SiLU+Mul
    int64_t inOffset = rowIdx * inColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    PackedVec quant_input;
    if constexpr (FUSE_SILU_MUL) {
      PackedVec in_vec_up =
          reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      quant_input = compute_silu_mul(in_vec, in_vec_up);
    } else {
      quant_input = in_vec;
    }

    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
        quant_input, SFScaleVal, sf_out);
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  // Grid, Block size.
  // Each thread converts 8 values.
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
  // Get number of blocks per SM
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x),
                     multiProcessorCount * numBlocksPerSM));
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

  int const blockRepeat =
      (totalWorkSize + block.x * grid.x - 1) / (block.x * grid.x);
  if (blockRepeat > 1) {
    size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
    // The shared-memory vectorized offset load only handles full 4-expert
    // chunks. Use the scalar specialization for the remainder cases.
    if (n_experts >= 4 && n_experts % 4 == 0) {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, false>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts);
    } else {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, true>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts);
    }
  } else {
    // The low-latency vectorized expert lookup only handles full 16-expert
    // chunks. Fall back to the scalar lookup path for the remainder cases.
    if (n_experts >= 16 && n_experts % 16 == 0) {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, false>
          <<<grid, block, 0, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, /* bool low_latency */ true);
    } else {
      cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false, true>
          <<<grid, block, 0, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, /* bool low_latency */ true);
    }
  }
}
#endif  // NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION

}  // namespace vllm

/*Quantization entry for fp4 experts quantization*/
#define CHECK_TH_CUDA(x, m) \
  STD_TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  STD_TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, m) \
  CHECK_TH_CUDA(x, m);    \
  CHECK_CONTIGUOUS(x, m);

constexpr auto HALF = torch::headeronly::ScalarType::Half;
constexpr auto BF16 = torch::headeronly::ScalarType::BFloat16;
constexpr auto FLOAT = torch::headeronly::ScalarType::Float;
constexpr auto INT = torch::headeronly::ScalarType::Int;
constexpr auto UINT8 = torch::headeronly::ScalarType::Byte;

// Common validation for fp4 experts quantization entry points.
static void validate_fp4_experts_quant_inputs(
    torch::stable::Tensor const& output,
    torch::stable::Tensor const& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts, int64_t m_topk,
    int64_t k) {
  CHECK_INPUT(output, "output");
  CHECK_INPUT(output_scale, "output_scale");
  CHECK_INPUT(input, "input");
  CHECK_INPUT(input_global_scale, "input_global_scale");
  CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts");
  CHECK_INPUT(output_scale_offset_by_experts, "output_scale_offset_by_experts");

  STD_TORCH_CHECK(output.dim() == 2);
  STD_TORCH_CHECK(output_scale.dim() == 2);
  STD_TORCH_CHECK(input.dim() == 2);
  STD_TORCH_CHECK(input_global_scale.dim() == 1);
  STD_TORCH_CHECK(input_offset_by_experts.dim() == 1);
  STD_TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  STD_TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  STD_TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  STD_TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  STD_TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
  // output is uint8 (two nvfp4 values are packed into one uint8)
  // output_scale is int32 (four fp8 values are packed into one int32)
  STD_TORCH_CHECK(output.scalar_type() == UINT8);
  STD_TORCH_CHECK(output_scale.scalar_type() == INT);

  const int BLOCK_SIZE = 16;
  STD_TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  STD_TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  STD_TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  STD_TORCH_CHECK(output.size(0) == m_topk);
  STD_TORCH_CHECK(output.size(1) == k / 2);
  int scales_k = k / BLOCK_SIZE;
  // 4 means the swizzle requirement by nvidia nvfp4.
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4 means 4 fp8 values are packed into one int32
  STD_TORCH_CHECK(output_scale.size(1) * 4 == padded_k);
}

void scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
  auto m_topk = input.size(0);
  auto k = input.size(1);

  validate_fp4_experts_quant_inputs(output, output_scale, input,
                                    input_global_scale, input_offset_by_experts,
                                    output_scale_offset_by_experts, m_topk, k);

  auto n_experts = input_global_scale.size(0);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "nvfp4_experts_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        vllm::quant_impl<cuda_type, /*FUSE_SILU_MUL=*/false>(
            output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
            input_global_scale.data_ptr(), input_offset_by_experts.data_ptr(),
            output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
            stream);
      });
}

void silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
  auto m_topk = input.size(0);
  // Input has gate || up layout, so k = input.size(1) / 2
  auto k_times_2 = input.size(1);
  STD_TORCH_CHECK(k_times_2 % 2 == 0, "input width must be even (gate || up)");
  auto k = k_times_2 / 2;

  validate_fp4_experts_quant_inputs(output, output_scale, input,
                                    input_global_scale, input_offset_by_experts,
                                    output_scale_offset_by_experts, m_topk, k);

  auto n_experts = input_global_scale.size(0);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_mul_nvfp4_experts_quant_kernel", [&] {
        using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
        vllm::quant_impl<cuda_type, /*FUSE_SILU_MUL=*/true>(
            output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
            input_global_scale.data_ptr(), input_offset_by_experts.data_ptr(),
            output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
            stream);
      });
}
