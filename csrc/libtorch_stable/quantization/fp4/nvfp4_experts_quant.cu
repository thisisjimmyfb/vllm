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
// SF swizzle tile: 128 rows x 4 SFs (64 columns), stored as 512 contiguous
// bytes.
#define CVT_FP4_SF_TILE_ROWS (128)
#define CVT_FP4_SF_TILE_COLS (64)
#define CVT_FP4_SF_TILE_BYTES (512)
// Expert boundaries the per-tile lookup steps over before switching to a
// bisection.
#define CVT_FP4_EXPERT_PROBE_STEPS (2)
// Fewest experts for which the per-expert tables are staged in shared memory.
// The lookup only bounds kernel time below one full wave, as a full wave keeps
// enough loads in flight to stay bandwidth-bound, and there it overlaps the
// first input load. From 32 experts input_offsets[n_experts] (the row count)
// and input_offsets[1] (the first probe) lie in different 128B lines, so
// reading from global costs two serial L2 round trips versus one parallel
// table load plus a barrier for staging. Below 32 both share a line, the probe
// hits L1 and staging only adds the barrier.
#define CVT_FP4_EXPERT_SMEM_MIN_EXPERTS (32)

namespace vllm {

template <class PackedVecT>
__device__ __forceinline__ PackedVecT LoadPackedVec(PackedVecT const* ptr) {
  if constexpr (VLLM_256B_PTX_ENABLED &&
                sizeof(PackedVecT) == sizeof(u32x8_t)) {
    PackedVecT vec;
    ld256(vec, ptr);
    return vec;
  } else {
    return *ptr;
  }
}

// Prefetch one packed vector into L1 without tying up registers.
template <class PackedVecT>
__device__ __forceinline__ void PrefetchPackedVecL1(PackedVecT const* ptr) {
  asm volatile("prefetch.global.L1 [%0];" ::"l"(ptr));
}

// Prefetch one packed vector into L2 without tying up registers; unlike an L1
// prefetch it also serves loads issued from other SMs.
template <class PackedVecT>
__device__ __forceinline__ void PrefetchPackedVecL2(PackedVecT const* ptr) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

// Store one 32B packed vector to global memory with streaming stores, as a
// single 256b store where it is available so each lane fills a whole sector.
template <class PackedVecT>
__device__ __forceinline__ void StorePackedVec(PackedVecT* ptr,
                                               PackedVecT const& vec) {
  static_assert(sizeof(PackedVecT) == sizeof(u32x8_t));
#if VLLM_256B_PTX_ENABLED
  st256_cs(reinterpret_cast<u32x8_t*>(ptr),
           reinterpret_cast<u32x8_t const&>(vec));
#else
  int4 const* const src = reinterpret_cast<int4 const*>(&vec);
  int4* const dst = reinterpret_cast<int4*>(ptr);
  st128_cs(dst, src[0]);
  st128_cs(dst + 1, src[1]);
#endif
}

// Entries between the per-expert shared tables: n_experts + 1 rounded up to 4
// so every table starts 16B aligned for the int4/float4 stores.
__host__ __device__ __forceinline__ int expert_table_stride(int n_experts) {
  return (n_experts + 1 + 3) & ~3;
}

// Shared bytes for the per-expert input offsets, SF offsets and global scales.
inline size_t expert_tables_smem_size(int n_experts) {
  return (2 * expert_table_stride(n_experts) + n_experts) * sizeof(uint32_t);
}

// Copy the per-expert input offsets and SF offsets (n_experts + 1 entries
// each) and global scales (n_experts entries) into shared memory. The global
// tables must be 16B aligned: whole 4-expert chunks are loaded vectorized and
// the remainder, including the [n_experts] sentinel, scalar, so any n_experts
// is supported.
__device__ __forceinline__ void load_expert_tables(
    uint32_t* __restrict__ shared_input_offsets,
    uint32_t* __restrict__ shared_output_scale_offsets,
    float* __restrict__ shared_SFScale,
    uint32_t const* __restrict__ input_offset_by_experts,
    uint32_t const* __restrict__ output_scale_offset_by_experts,
    float const* __restrict__ SFScale, int n_experts) {
  float4 const ones = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  int const tail_begin = n_experts & ~3;
  int i = threadIdx.x * 4;
  while (i < tail_begin) {
    *reinterpret_cast<int4*>(&shared_input_offsets[i]) =
        *reinterpret_cast<int4 const*>(&input_offset_by_experts[i]);
    *reinterpret_cast<int4*>(&shared_output_scale_offsets[i]) =
        *reinterpret_cast<int4 const*>(&output_scale_offset_by_experts[i]);
    *reinterpret_cast<float4*>(&shared_SFScale[i]) =
        SFScale == nullptr ? ones
                           : *reinterpret_cast<float4 const*>(&SFScale[i]);
    i += blockDim.x * 4;
  }
  i = tail_begin + threadIdx.x;
  while (i < n_experts) {
    shared_input_offsets[i] = input_offset_by_experts[i];
    shared_output_scale_offsets[i] = output_scale_offset_by_experts[i];
    shared_SFScale[i] = SFScale == nullptr ? 1.0f : SFScale[i];
    i += blockDim.x;
  }
  if (i <= n_experts) {
    shared_input_offsets[i] = input_offset_by_experts[i];
    shared_output_scale_offsets[i] = output_scale_offset_by_experts[i];
  }
}

// Walk expert_idx forward to the last expert starting at or before row; the
// caller's rows only advance. Crossing none or one boundary is the usual case,
// so a few steps are probed before a grid stride's jump over many experts is
// bisected. Empty experts share their offset with the next one and are
// stepped over.
__device__ __forceinline__ int search_expert(uint32_t const* input_offsets,
                                             int n_experts, int expert_idx,
                                             uint32_t row) {
  int probes = 0;
  while (probes < CVT_FP4_EXPERT_PROBE_STEPS &&
         input_offsets[expert_idx + 1] <= row) {
    ++expert_idx;
    ++probes;
  }
  if (probes >= CVT_FP4_EXPERT_PROBE_STEPS &&
      input_offsets[expert_idx + 1] <= row) {
    int lo = expert_idx + 1, hi = n_experts - 1;
    while (lo < hi) {
      int const mid = (lo + hi + 1) >> 1;
      bool const le = input_offsets[mid] <= row;
      lo = le ? mid : lo;
      hi = le ? hi : mid - 1;
    }
    expert_idx = lo;
  }
  return expert_idx;
}

// Count the experts starting at or before a warp-uniform row, so the row's
// expert is the count minus 1; empty experts share the next one's offset and
// are counted, which steps over them. Each lane compares 4 consecutive offsets
// per 128 experts and the warp's votes sum them: one 16B shared load per 128
// experts and no dependent steps, where search_expert walks the table
// serially. The whole warp must call this; the table must be 16B aligned.
__device__ __forceinline__ int warp_count_experts(uint32_t const* input_offsets,
                                                  int n_experts, uint32_t row) {
  int count = 0;
#pragma unroll 1
  for (int base = 0; base < n_experts; base += 4 * 32) {
    // Lanes past the last expert read the first entries and mask them all.
    int const j = base + threadIdx.x % 32 * 4;
    uint4 o =
        *reinterpret_cast<uint4 const*>(&input_offsets[j < n_experts ? j : 0]);
    o.x = j < n_experts ? o.x : ~0u;
    o.y = j + 1 < n_experts ? o.y : ~0u;
    o.z = j + 2 < n_experts ? o.z : ~0u;
    o.w = j + 3 < n_experts ? o.w : ~0u;
    count += __popc(__ballot_sync(~0u, o.x <= row)) +
             __popc(__ballot_sync(~0u, o.y <= row)) +
             __popc(__ballot_sync(~0u, o.z <= row)) +
             __popc(__ballot_sync(~0u, o.w <= row));
  }
  return count;
}

// Block and grid sizes for one wave of blocks grid-striding over work_items,
// one per thread. The block shrinks from the largest size while its blocks
// cannot fill every resident block slot, so small inputs still spread over
// all SMs, but stops at the minimum block size or before exceeding one wave
// of full-size blocks. Only multiples of the warp size dividing the SM's
// thread capacity are taken, so the resident blocks fill it exactly.
template <class Kernel>
std::pair<int, int> one_wave_launch_dims(Kernel kernel, int64_t work_items,
                                         size_t shared_mem_size) {
  static constexpr int kWarpSize = 32;
  // Each block stages the expert tables once, so a block keeps at least this
  // many threads to share that cost.
  static constexpr int kMinThreadsPerBlock = 128;
  int const multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  // Resident block slots on the device for a block of the given size.
  int numBlocksPerSM = 0;
  int threads = CVT_FP4_MAX_THREADS_PER_BLOCK;
  STD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, kernel, threads, shared_mem_size));
  int64_t block_slots =
      (int64_t)multiProcessorCount * std::max(numBlocksPerSM, 1);
  int64_t const max_blocks = block_slots;

  // Queried directly: get_device_attribute caches the first attribute it is
  // asked for (here the SM count) and returns it for every other one.
  int device = 0;
  STD_CUDA_CHECK(cudaGetDevice(&device));
  int maxThreadsPerSM = 0;
  STD_CUDA_CHECK(cudaDeviceGetAttribute(
      &maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  while (threads > kMinThreadsPerBlock &&
         div_round_up(work_items, (int64_t)threads) < block_slots) {
    int next_threads = threads - kWarpSize;
    while (next_threads > kMinThreadsPerBlock &&
           maxThreadsPerSM % next_threads != 0) {
      next_threads -= kWarpSize;
    }
    if (next_threads < kMinThreadsPerBlock ||
        maxThreadsPerSM % next_threads != 0 ||
        div_round_up(work_items, (int64_t)next_threads) > max_blocks) {
      break;
    }
    threads = next_threads;
    STD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, kernel, threads, shared_mem_size));
    block_slots = (int64_t)multiProcessorCount * std::max(numBlocksPerSM, 1);
  }
  int const blocks = static_cast<int>(std::min<int64_t>(
      div_round_up(work_items, (int64_t)threads), block_slots));
  return {threads, blocks};
}

// NVFP4 quantization kernel for LARGE_M_TOPK = true (large m_topk optimized
// version). When FUSE_SILU_MUL=true, expects input with gate||up layout and
// fuses SiLU(gate)*up before quantization.
#if NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION == 3
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false>
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
  static_assert(
      CVT_FP4_NUM_THREADS_PER_SF == 1,
      "CVT_FP4_NUM_THREADS_PER_SF should be 1 with NVFP4_ENABLE_ELTS16");
  static constexpr int kVecsPerKTile =
      CVT_FP4_SF_TILE_COLS / CVT_FP4_ELTS_PER_THREAD;
  static_assert(kVecsPerKTile == 4);
  static constexpr int kTileVecs = CVT_FP4_SF_TILE_ROWS * kVecsPerKTile;
  static_assert(CVT_FP4_MAX_THREADS_PER_BLOCK == kTileVecs,
                "the largest block takes one vec of a tile per thread");
  static constexpr int kInnerRowStride = CVT_FP4_SF_TILE_ROWS / 4;

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;
  int const colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int const inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  PackedVec const* const in_vecs = reinterpret_cast<PackedVec const*>(in);

  uint32_t const* input_offsets = input_offset_by_experts;
  uint32_t const* output_scale_offset = output_scale_offset_by_experts;
  float const* sf_scales = SFScale;

  // Every thread looks up an expert for every tile it works on, so the
  // per-expert tables are staged in shared memory once per block; reading
  // them from global memory would issue more sectors than the tile's input.
  {
    extern __shared__ __align__(16) uint32_t shared_expert_tables[];
    int const table_stride = expert_table_stride(n_experts);

    uint32_t* input_offsets_table = shared_expert_tables;
    uint32_t* output_offsets_table = shared_expert_tables + table_stride;
    float* sf_table =
        reinterpret_cast<float*>(shared_expert_tables + 2 * table_stride);

    load_expert_tables(input_offsets_table, output_offsets_table, sf_table,
                       input_offset_by_experts, output_scale_offset_by_experts,
                       SFScale, n_experts);

    input_offsets = input_offsets_table;
    output_scale_offset = output_offsets_table;
    sf_scales = sf_table;

    // Warm L2 with the first round's input while the tables arrive, so the
    // first loads overlap the tables' fetch instead of following it. It is
    // issued after the tables' loads: issued before them, the whole round's
    // prefetches queue ahead of the tables' few sectors and delay the barrier
    // by more than they save. Its rows need no tables: a tile's rotation only
    // permutes its rows among its vecs, so taking them unrotated still covers
    // every vec of the round, each from some thread of the grid. The valid
    // row count lives in the tables, so numRows bounds it instead.
    {
      int64_t const tileVec = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
      int const tile = static_cast<int>(tileVec / kTileVecs);
      int const t = static_cast<int>(tileVec % kTileVecs);
      int const rowIdx = tile / numKTiles * CVT_FP4_SF_TILE_ROWS +
                         (t % 16) / kVecsPerKTile * kInnerRowStride + t / 16;
      int const colIdx = tile % numKTiles * kVecsPerKTile + t % kVecsPerKTile;
      if (rowIdx < numRows && colIdx < colsPerRow) {
        int64_t const inOffset = (int64_t)rowIdx * inColsPerRow + colIdx;
        PrefetchPackedVecL2(in_vecs + inOffset);
        if constexpr (FUSE_SILU_MUL) {
          PrefetchPackedVecL2(in_vecs + colsPerRow + inOffset);
        }
      }
    }

    __syncthreads();
  }

  int const numValidRows =
      min(numRows, static_cast<int>(input_offsets[n_experts]));
  int const numMTiles = div_round_up(numValidRows, CVT_FP4_SF_TILE_ROWS);
  int const numTiles = numMTiles * numKTiles;
  // Every tile's vecs, vec t in 0..kTileVecs-1 of each, in one flat range
  // that blocks take blockDim vecs at a time. The block size is a multiple of
  // the warp size, so a warp's vecs never straddle two tiles and it still
  // writes one whole SF sector; a block may.
  int64_t const numTileVecs = (int64_t)numTiles * kTileVecs;

  // K chunks are the fast tile dimension, so blocks working at the same time
  // read consecutive 128B segments of the same rows.
  for (int64_t tileVec = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       tileVec < numTileVecs; tileVec += (int64_t)gridDim.x * blockDim.x) {
    int const tile = static_cast<int>(tileVec / kTileVecs);
    int const colBegin = tile % numKTiles * kVecsPerKTile;
    int const rowBegin = tile / numKTiles * CVT_FP4_SF_TILE_ROWS;
    int const tileRows = min(CVT_FP4_SF_TILE_ROWS, numValidRows - rowBegin);

    // Vec t of a tile writes byte t of the SF tile, so each warp writes one
    // 32B sector. An SF tile stores row r (0..127) and K column innerKIdx
    // (0..3) at byte
    //   s = (r % 32) * 16 + (r / 32) * 4 + innerKIdx,
    // a mixed-radix number, so s = t splits into innerKIdx = t % 4,
    // r / 32 = (t % 16) / 4 and r % 32 = t / 16, giving the row
    //   r = ((t % 16) / 4) * 32 + t / 16.
    // r is a row within an SF tile, i.e. a row within the expert mod 128;
    // each tile takes its row congruent to it (see rowInTile).
    int const t = static_cast<int>(tileVec % kTileVecs);
    int const rowInSFTile = (t % 16) / kVecsPerKTile * kInnerRowStride + t / 16;
    int const innerKIdx = t % kVecsPerKTile;

    // A warp never straddles two tiles, so rowBegin is warp-uniform.
    int const tile_expert_idx =
        warp_count_experts(input_offsets, n_experts,
                           static_cast<uint32_t>(rowBegin)) -
        1;
    int const firstRowInExpert = rowBegin - input_offsets[tile_expert_idx];
    int const rowInTile =
        (rowInSFTile - firstRowInExpert) & (CVT_FP4_SF_TILE_ROWS - 1);
    int const colIdx = colBegin + innerKIdx;
    // The last tile row and the last K chunk can be partial.
    if (rowInTile >= tileRows || colIdx >= colsPerRow) {
      continue;
    }
    int const rowIdx = rowBegin + rowInTile;
    int64_t const inOffset = (int64_t)rowIdx * inColsPerRow + colIdx;

    // The input is loaded before the row's expert lookup so the lookup
    // overlaps it.
    PackedVec in_vec = LoadPackedVec(in_vecs + inOffset);
    PackedVec in_vec_up;
    if constexpr (FUSE_SILU_MUL) {
      in_vec_up = LoadPackedVec(in_vecs + colsPerRow + inOffset);
    }

    // A thread's vecs within one tile do not visit rows in order, so the
    // row's expert is walked from the tile's first expert rather than carried.
    int const expert_idx =
        search_expert(input_offsets, n_experts, tile_expert_idx,
                      static_cast<uint32_t>(rowIdx));
    int const rowIdx_in_expert = rowIdx - input_offsets[expert_idx];
    float const SFScaleVal = sf_scales[expert_idx];
    uint32_t* const SFout_in_expert =
        SFout + output_scale_offset[expert_idx] * numKTiles;

    // Optionally apply fused SiLU+Mul
    if constexpr (FUSE_SILU_MUL) {
      in_vec = compute_silu_mul(in_vec, in_vec_up);
    }

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    u32x2 const o =
        cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
            in_vec, SFScaleVal, sf_out);

    uint64_t* const out_row =
        reinterpret_cast<uint64_t*>(out) + (int64_t)rowIdx * colsPerRow;
    st64_cs(reinterpret_cast<int64_t*>(out_row + colIdx),
            static_cast<int64_t>(static_cast<uint64_t>(o.hi) << 32 | o.lo));
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  if (m_topk == 0) return;

  // A whole tile is one vec per thread of the largest block. Blocks take any
  // multiple of the warp size of the tiles' vecs, as each warp writes whole
  // SF sectors.
  static constexpr int kTileVecs = CVT_FP4_MAX_THREADS_PER_BLOCK;
  int64_t const max_tile_vecs =
      div_round_up((int64_t)m_topk, (int64_t)CVT_FP4_SF_TILE_ROWS) *
      div_round_up(k, CVT_FP4_SF_TILE_COLS) * kTileVecs;
  size_t const shared_mem_size = expert_tables_smem_size(n_experts);
  auto const kernel = cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false>;
  auto const [threads, blocks] =
      one_wave_launch_dims(kernel, max_tile_vecs, shared_mem_size);

  kernel<<<blocks, threads, shared_mem_size, stream>>>(
      m_topk, k, reinterpret_cast<T*>(input),
      reinterpret_cast<float*>(input_global_scale),
      reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(output_scale),
      reinterpret_cast<uint32_t*>(input_offset_by_experts),
      reinterpret_cast<uint32_t*>(output_scale_offset_by_experts), n_experts);
}
#elif NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION == 2
// Blocks grid-stride over tiles of 128 consecutive input rows by one
// 64-column K chunk. Tiles ignore expert boundaries, so every tile but the last
// row of tiles is full however small the experts are, and each thread looks up
// the expert of its own row for every tile. Padding rows are never visited.
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false>
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
  static_assert(
      CVT_FP4_NUM_THREADS_PER_SF == 1,
      "CVT_FP4_NUM_THREADS_PER_SF should be 1 with NVFP4_ENABLE_ELTS16");
  static constexpr int kVecsPerKTile =
      CVT_FP4_SF_TILE_COLS / CVT_FP4_ELTS_PER_THREAD;
  static_assert(kVecsPerKTile == 4);

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;
  int const colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int const inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  PackedVec const* const in_vecs = reinterpret_cast<PackedVec const*>(in);

  // Every thread looks up an expert for every tile it works on, so the
  // per-expert tables are staged in shared memory once per block; reading
  // them from global memory would issue more sectors than the tile's input.
  extern __shared__ __align__(16) uint32_t shared_expert_tables[];
  int const table_stride = expert_table_stride(n_experts);
  uint32_t* const input_offsets = shared_expert_tables;
  uint32_t* const output_scale_offset = shared_expert_tables + table_stride;
  float* const sf_scales =
      reinterpret_cast<float*>(shared_expert_tables + 2 * table_stride);
  load_expert_tables(input_offsets, output_scale_offset, sf_scales,
                     input_offset_by_experts, output_scale_offset_by_experts,
                     SFScale, n_experts);
  __syncthreads();

  // The tile's SFs, one 32-bit word (a row's four K SFs) per row, each row's
  // destination word in SFout, and the first row's row within its expert.
  // Row r is kept in slot (r % 32) * 4 + r / 32, the SF tile's own order: the
  // write-out reads rows 32 apart from adjacent lanes, which in row order
  // would share a bank.
  __shared__ uint32_t tile_sf_words[CVT_FP4_SF_TILE_ROWS];
  __shared__ uint32_t tile_sf_dst[CVT_FP4_SF_TILE_ROWS];
  __shared__ int tile_first_row_in_expert;
  uint8_t* const tile_sf_bytes = reinterpret_cast<uint8_t*>(tile_sf_words);
  static constexpr int kInnerRowStride = CVT_FP4_SF_TILE_ROWS / 4;

  int const numValidRows =
      min(numRows, static_cast<int>(input_offsets[n_experts]));
  int const numMTiles = div_round_up(numValidRows, CVT_FP4_SF_TILE_ROWS);
  int const numTiles = numMTiles * numKTiles;

  // A thread's rows only advance, so its expert is carried across tiles.
  int expert_idx = 0;

  // K chunks are the fast tile dimension, so blocks working at the same time
  // read consecutive 128B segments of the same rows.
  for (int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {
    int const colBegin = tile % numKTiles * kVecsPerKTile;
    int const rowBegin = tile / numKTiles * CVT_FP4_SF_TILE_ROWS;
    int const tileRows = min(CVT_FP4_SF_TILE_ROWS, numValidRows - rowBegin);
    int const tileVecs = tileRows * kVecsPerKTile;

    for (int idx = threadIdx.x; idx < tileVecs; idx += blockDim.x) {
      int const rowInTile = idx / kVecsPerKTile;
      int const colIdx = colBegin + idx % kVecsPerKTile;
      // The last K chunk is partial when K is not a multiple of 64.
      if (colIdx >= colsPerRow) {
        continue;
      }
      int const rowIdx = rowBegin + rowInTile;
      int64_t const inOffset = (int64_t)rowIdx * inColsPerRow + colIdx;

      // The input is loaded before the expert lookup so the lookup overlaps it.
      PackedVec in_vec = LoadPackedVec(in_vecs + inOffset);
      PackedVec in_vec_up;
      if constexpr (FUSE_SILU_MUL) {
        in_vec_up = LoadPackedVec(in_vecs + colsPerRow + inOffset);
      }

      expert_idx = search_expert(input_offsets, n_experts, expert_idx,
                                 static_cast<uint32_t>(rowIdx));
      int const rowIdx_in_expert = rowIdx - input_offsets[expert_idx];
      float const SFScaleVal = sf_scales[expert_idx];
      uint32_t* const SFout_in_expert =
          SFout + output_scale_offset[expert_idx] * numKTiles;

      // Optionally apply fused SiLU+Mul
      if constexpr (FUSE_SILU_MUL) {
        in_vec = compute_silu_mul(in_vec, in_vec_up);
      }

      uint8_t sf;
      u32x2 const o =
          cvt_warp_fp16_to_fp4_and_sf<Type, CVT_FP4_NUM_THREADS_PER_SF,
                                      UE8M0_SF>(in_vec, SFScaleVal, sf);
      int const slot =
          (rowInTile % kInnerRowStride) * 4 + rowInTile / kInnerRowStride;
      tile_sf_bytes[slot * kVecsPerKTile + idx % kVecsPerKTile] = sf;
      if (idx % kVecsPerKTile == 0) {
        // colIdx starts the K chunk, so this is the row's word-aligned slot.
        uint8_t const* const sf_out =
            cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                               CVT_FP4_NUM_THREADS_PER_SF>(
                rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);
        tile_sf_dst[slot] = static_cast<uint32_t>(
            (sf_out - reinterpret_cast<uint8_t const*>(SFout)) /
            sizeof(uint32_t));
        if (idx == 0) {
          tile_first_row_in_expert = rowIdx_in_expert;
        }
      }

      uint64_t* const out_row =
          reinterpret_cast<uint64_t*>(out) + (int64_t)rowIdx * colsPerRow;
      st64_cs(reinterpret_cast<int64_t*>(out_row + colIdx),
              static_cast<int64_t>(static_cast<uint64_t>(o.hi) << 32 | o.lo));
    }
    __syncthreads();

    // Write the tile's SFs a word per thread. In an SF tile, the word of row
    // r sits at (r % 32) * 4 + (r / 32) % 4, so thread j takes the row whose
    // row within its expert is congruent to (j % 4) * 32 + j / 4 modulo the
    // tile height. Within one expert, consecutive threads then write
    // consecutive words of at most two SF tiles; across experts this is still
    // a permutation of the tile's rows.
    if (threadIdx.x < CVT_FP4_SF_TILE_ROWS) {
      int const j = threadIdx.x;
      int const rowInSFTile = (j % 4) * kInnerRowStride + j / 4;
      int const rowInTile =
          (rowInSFTile - tile_first_row_in_expert) & (CVT_FP4_SF_TILE_ROWS - 1);
      int const slot =
          (rowInTile % kInnerRowStride) * 4 + rowInTile / kInnerRowStride;
      int const chunkCols = min(kVecsPerKTile, colsPerRow - colBegin);
      if (rowInTile < tileRows) {
        if (chunkCols == kVecsPerKTile) {
          SFout[tile_sf_dst[slot]] = tile_sf_words[slot];
        } else {
          // A partial K chunk writes only its real SFs.
          uint8_t* const sf_row =
              reinterpret_cast<uint8_t*>(SFout + tile_sf_dst[slot]);
          for (int innerKIdx = 0; innerKIdx < chunkCols; ++innerKIdx) {
            sf_row[innerKIdx] = tile_sf_bytes[slot * kVecsPerKTile + innerKIdx];
          }
        }
      }
    }
    // The next tile reuses the staging buffers.
    __syncthreads();
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  if (m_topk == 0) return;

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  // One wave of blocks grid-striding over the tiles, capped at the tile count.
  int64_t const max_tiles =
      div_round_up((int64_t)m_topk, (int64_t)CVT_FP4_SF_TILE_ROWS) *
      div_round_up(k, CVT_FP4_SF_TILE_COLS);
  dim3 block(CVT_FP4_MAX_THREADS_PER_BLOCK);
  size_t const shared_mem_size = expert_tables_smem_size(n_experts);
  auto const kernel = cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false>;
  int numBlocksPerSM = 0;
  STD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSM, kernel, block.x, shared_mem_size));
  dim3 grid(std::min<int64_t>(
      max_tiles, (int64_t)multiProcessorCount * std::max(numBlocksPerSM, 1)));

  kernel<<<grid, block, shared_mem_size, stream>>>(
      m_topk, k, reinterpret_cast<T*>(input),
      reinterpret_cast<float*>(input_global_scale),
      reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(output_scale),
      reinterpret_cast<uint32_t*>(input_offset_by_experts),
      reinterpret_cast<uint32_t*>(output_scale_offset_by_experts), n_experts);
}
#elif NVFP4_EXPERTS_QUANT_LOCAL_OPTIMIZATION == 1
template <class Type, bool FUSE_SILU_MUL = false, bool UE8M0_SF = false>
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
  static_assert(
      CVT_FP4_NUM_THREADS_PER_SF == 1,
      "CVT_FP4_NUM_THREADS_PER_SF should be 1 with NVFP4_ENABLE_ELTS16");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;
  int const colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;

  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const gridStride = gridDim.x * blockDim.x;
  // When fusing SiLU+Mul, input has gate || up layout (doubled width)
  int const inColsPerRow = FUSE_SILU_MUL ? colsPerRow * 2 : colsPerRow;

  PackedVec const* const in_vecs = reinterpret_cast<PackedVec const*>(in);

  // Warm L1 with the first element's input while the tables are staged. The
  // valid row count lives in the tables, so this is bounded by numRows.
  if (tid < numRows * colsPerRow) {
    int64_t const inOffset =
        (int64_t)(tid / colsPerRow) * inColsPerRow + tid % colsPerRow;
    PrefetchPackedVecL1(in_vecs + inOffset);
    if constexpr (FUSE_SILU_MUL) {
      PrefetchPackedVecL1(in_vecs + colsPerRow + inOffset);
    }
  }

  // Per-expert tables, read from global memory unless staged in shared memory
  // (see CVT_FP4_EXPERT_SMEM_MIN_EXPERTS).
  uint32_t const* input_offsets = input_offset_by_experts;
  uint32_t const* output_scale_offset = output_scale_offset_by_experts;
  float const* sf_scales = SFScale;
  if (n_experts >= CVT_FP4_EXPERT_SMEM_MIN_EXPERTS) {
    extern __shared__ __align__(16) uint32_t shared_expert_tables[];
    int const table_stride = expert_table_stride(n_experts);
    uint32_t* const shared_input_offsets = shared_expert_tables;
    uint32_t* const shared_sf_offsets = shared_expert_tables + table_stride;
    float* const shared_SFScale =
        reinterpret_cast<float*>(shared_expert_tables + 2 * table_stride);
    load_expert_tables(shared_input_offsets, shared_sf_offsets, shared_SFScale,
                       input_offset_by_experts, output_scale_offset_by_experts,
                       SFScale, n_experts);
    __syncthreads();
    input_offsets = shared_input_offsets;
    output_scale_offset = shared_sf_offsets;
    sf_scales = shared_SFScale;
  }

  int32_t const numValidRows =
      min(numRows, static_cast<int32_t>(input_offsets[n_experts]));
  int const numValidVecs = numValidRows * colsPerRow;

  // A thread's rows only advance, so its expert is carried across iterations.
  int expert_idx = 0;

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numValidVecs; globalIdx += gridStride) {
    // The input is loaded before the expert lookup so the lookup overlaps it.
    PackedVec in_vec, in_vec_up;
    if constexpr (FUSE_SILU_MUL) {
      int64_t const inOffset =
          (int64_t)(globalIdx << 1) - (globalIdx % colsPerRow);
      in_vec = LoadPackedVec(in_vecs + inOffset);
      in_vec_up = LoadPackedVec(in_vecs + colsPerRow + inOffset);
    } else {
      in_vec = LoadPackedVec(in_vecs + globalIdx);
    }

    // Walk the expert forward a few steps; crossing none or one boundary is
    // the usual case. Empty experts share their offset with the next one and
    // are stepped over.
    uint32_t rowIdx = globalIdx / colsPerRow;
    uint32_t colIdx = globalIdx % colsPerRow;

    int probes = 0;
    while (probes < CVT_FP4_EXPERT_PROBE_STEPS &&
           input_offsets[expert_idx + 1] <= rowIdx) {
      ++expert_idx;
      ++probes;
    }
    // A grid stride can jump over many experts; bisect the rest.
    // input_offsets[expert_idx + 1] <= row < input_offsets[
    // n_experts], so the last expert starting at or before row lies in
    // [expert_idx + 1, n_experts - 1].
    if (probes >= CVT_FP4_EXPERT_PROBE_STEPS &&
        input_offsets[expert_idx + 1] <= rowIdx) {
      int lo = expert_idx + 1, hi = n_experts - 1;
      while (lo < hi) {
        int const mid = (lo + hi + 1) >> 1;
        bool const le = input_offsets[mid] <= rowIdx;
        lo = le ? mid : lo;
        hi = le ? hi : mid - 1;
      }
      expert_idx = lo;
    }
    uint32_t const rowIdx_in_expert = rowIdx - input_offsets[expert_idx];

    // Optionally apply fused SiLU+Mul
    if constexpr (FUSE_SILU_MUL) {
      in_vec = compute_silu_mul(in_vec, in_vec_up);
    }

    uint64_t* out_row =
        reinterpret_cast<uint64_t*>(out) +
        (int64_t)rowIdx * (colsPerRow / CVT_FP4_NUM_THREADS_PER_SF);

    float const SFScaleVal =
        sf_scales == nullptr ? 1.0f : sf_scales[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    u32x2 const o =
        cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>(
            in_vec, SFScaleVal, sf_out);

    st64_cs(reinterpret_cast<int64_t*>(out_row + colIdx),
            static_cast<int64_t>(static_cast<uint64_t>(o.hi) << 32 | o.lo));
  }
}

template <typename T, bool FUSE_SILU_MUL = false>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, cudaStream_t stream) {
  if (m_topk == 0) return;

  int64_t const total_thread_work = (int64_t)m_topk * (k / ELTS_PER_THREAD);
  size_t const shared_mem_size = n_experts >= CVT_FP4_EXPERT_SMEM_MIN_EXPERTS
                                     ? expert_tables_smem_size(n_experts)
                                     : 0;
  auto const kernel = cvt_fp16_to_fp4<T, FUSE_SILU_MUL, false>;
  auto const [threads, blocks] =
      one_wave_launch_dims(kernel, total_thread_work, shared_mem_size);

  kernel<<<blocks, threads, shared_mem_size, stream>>>(
      m_topk, k, reinterpret_cast<T*>(input),
      reinterpret_cast<float*>(input_global_scale),
      reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(output_scale),
      reinterpret_cast<uint32_t*>(input_offset_by_experts),
      reinterpret_cast<uint32_t*>(output_scale_offset_by_experts), n_experts);
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
  // The kernel addresses output_scale by 32-bit byte offsets.
  STD_TORCH_CHECK(output_scale.numel() * 4 <= UINT32_MAX,
                  "output_scale must be smaller than 4GB");
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
