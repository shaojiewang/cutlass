/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined Implicit GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize_ = Conv2dProblemSize,  ///! Convolutional operator on 2D or 3D problem
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone    ///! Group mode
>
struct ImplicitGemmConvolutionStreamK {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm; 
  static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;

  /// The per-thread tile of raw accumulators
  using AccumulatorTile = typename Mma::FragmentC;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Workspace bytes per thread block
  static size_t const kWorkspaceBytesPerBlock =
    __NV_STD_MAX(
      kThreadCount * sizeof(AccumulatorTile),
      Epilogue::kWorkspaceBytesPerBlock);

  /// Block-striped reduction utility
  using BlockStripedReduceT = BlockStripedReduce<kThreadCount, AccumulatorTile>;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  /// Check iterator A and B convolution dimension are the same and 
  // set device::ImplicitGemmConvolutionStreamK::kConvDim
  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim, 
    "Convolution on different different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  static conv::GroupMode const kGroupMode = GroupMode_;

  /// Wgrad C stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix C (KxRSC) 
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx = 
    platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

  //
  //
  //
  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout, 
    TensorRefC,
    ConvOperator,
    ConvProblemSize
    >;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmUniversalMode mode;

    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;
    TensorRefC ref_D;
    typename EpilogueOutputOp::Params output_op;
    SplitKMode split_k_mode;

    int avail_sms;          /// The number of SMs that StreamK dispatch heuristics will attempt to load-balance across (-1 defaults to device width, 1 implies classic data-parallel scheduling)


    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      ConvProblemSize const & problem_size
    ):
      problem_size(problem_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size,
      TensorRefA const & ref_A,
      TensorRefB const & ref_B,
      TensorRefC const & ref_C,
      TensorRefC const & ref_D,
      typename EpilogueOutputOp::Params const & output_op,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size(problem_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      ref_D(ref_D),
      output_op(output_op),
      split_k_mode(split_k_mode)
    {

    }

  };

  /// Parameters structure
  struct Params {

  public:
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    //int swizzle_log_tile;

    int gemm_k_iterations;
    int gemm_k_iterations_per_channel;
    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const *ptr_A;
    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const *ptr_B;
    typename Epilogue::OutputTileIterator::Params iterator_C;
    typename Epilogue::OutputTileIterator::Element *ptr_C;
    typename Epilogue::OutputTileIterator::Params iterator_D;
    typename Epilogue::OutputTileIterator::Element *ptr_D;
    typename EpilogueOutputOp::Params output_op;
    int *semaphore;
    SplitKMode split_k_mode;

    ThreadblockSwizzle block_mapping;

    void *barrier_workspace;
    void *partials_workspace;

    //
    // Methods
    //

  protected:
    /// Pad the given allocation size up to the nearest cache line
    static size_t cacheline_align_up(size_t size)
    {
      static const int CACHELINE_SIZE = 128;
      return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
    }

    /// Get the workspace size needed for barrier
    size_t get_barrier_workspace_size() const
    {
      // For atomic reduction, each SK-block needs a synchronization flag.  For parallel reduction,
      // each reduction block needs its own synchronization flag.
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      int num_flags = fast_max(sk_blocks, block_mapping.reduction_blocks);

      return cacheline_align_up(sizeof(typename Barrier::T) * num_flags);
    }

    /// Get the workspace size needed for intermediate partial sums
    size_t get_partials_workspace_size() const
    {
      int sk_blocks = block_mapping.sk_regions() * block_mapping.sk_blocks_per_region();
      return cacheline_align_up(kWorkspaceBytesPerBlock * sk_blocks);
    }

  public:

    CUTLASS_HOST_DEVICE
    Params(): gemm_k_iterations(0) { }

    /// 
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      iterator_C(ConvOutputIteratorParameter::layout(args.ref_C)),
      ptr_C(args.ref_C.data()),
      iterator_D(ConvOutputIteratorParameter::layout(args.ref_D)),
      ptr_D(args.ref_D.data()),
      output_op(args.output_op),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(
        kConvolutionalOperator,
        ThreadblockShape::kK,
        args.problem_size,
        kIteratorAlgorithm,
        kGroupMode,
        ThreadblockShape::kN);

      gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
          kConvolutionalOperator, ThreadblockShape::kK, args.problem_size, kIteratorAlgorithm);

      // Number of SMs to make available for StreamK decomposition
      int avail_sms = (args.avail_sms == -1) ?
                        device_sms :
                        fast_min(args.avail_sms, device_sms);

      // Initialize the block mapping structure
      block_mapping = ThreadblockSwizzle(
        typename ThreadblockSwizzle::template KernelTraits<ImplicitGemmConvolutionStreamK>(),
        GemmUniversalMode::kBatched,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        1,
        sm_occupancy,
        device_sms,
        avail_sms);

      grid_tiled_shape = block_mapping.tiled_shape();

    }

    /// Returns the workspace size (in bytes) needed for these parameters
    size_t get_workspace_size() const
    {
      return
        get_barrier_workspace_size() +
        get_partials_workspace_size();
    }


    /// Assign and initialize the specified workspace buffer.  Assumes
    /// the memory allocated to workspace is at least as large as get_workspace_size().
    Status init_workspace(
      void *workspace,
      cudaStream_t stream = nullptr)
    {
      uint8_t *ptr = static_cast<uint8_t*>(workspace);

      // Establish partials workspace
      partials_workspace = nullptr;
      size_t partials_workspace_bytes = get_partials_workspace_size();
      if (partials_workspace_bytes > 0)
      {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        partials_workspace = ptr;
        ptr += partials_workspace_bytes;
      }

      // Establish barrier workspace
      barrier_workspace = nullptr;
      size_t barrier_workspace_bytes = get_barrier_workspace_size();
      if (barrier_workspace_bytes > 0)
      {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }
        barrier_workspace = ptr;
        ptr += barrier_workspace_bytes;
      }

      // Zero-initialize barrier workspace
      if (barrier_workspace)
      {
        size_t barrier_workspace_bytes = get_barrier_workspace_size();

        CUTLASS_TRACE_HOST("  Initialize " << barrier_workspace_bytes << " barrier bytes");

        cudaError_t result = cudaMemsetAsync(
          barrier_workspace,
          0,
          barrier_workspace_bytes,
          stream);

        if (result != cudaSuccess) {
          CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }

      return Status::kSuccess;
    }


    /// Returns the GEMM volume in thread block tiles
    cutlass::gemm::GemmCoord get_tiled_shape() const
    {
      return block_mapping.tiled_shape();
    }


    /// Returns the total number of thread blocks to launch
    int get_grid_blocks() const
    {
      dim3 grid_dims = get_grid_dims();
      return grid_dims.x * grid_dims.y * grid_dims.z;
    }


    /// Returns the grid extents in thread blocks to launch
    dim3 get_grid_dims() const
    {
      return block_mapping.get_grid_dims();
    }


    /// Lightweight update given a subset of arguments.
    void update(Arguments const &args)
    {
      CUTLASS_TRACE_HOST("GemmUniversalStreamK::Params::update()");

      // Update input/output pointers
      
    }
  };

  /// Tile work descriptor
  struct TileWorkDesc
  {
    /// The linear tile index
    int tile_idx;

    /// The location of this tile (in threadblock-tile coordinates) in the output matrix
    cutlass::gemm::GemmCoord tiled_coord;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    int iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_begin;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    int k_end;

    /// The number of remaining MAC-iterations this threadblock will perform for this tile
    int k_iters_remaining;

    // Whether this block will perform the first iteration of this tile
    CUTLASS_DEVICE
    bool tile_started()
    {
      return (k_begin == 0);
    }

    // Whether this block will perform the last iteration of this tile
    CUTLASS_DEVICE
    bool tile_finished(Params const &params)
    {
      return (k_end == params.block_mapping.problem_size.k());
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

protected:

  //
  // Data members
  //

  /// GEMM problem parameters
  Params params;

  /// Shared storage reference
  SharedStorage &shared_storage;

  /// ID within the threadblock
  int thread_idx;

  /// ID of warp
  int warp_idx;

  /// ID of each thread within a warp
  int lane_idx;

  /// Threadblock scoped epilogue
  Epilogue epilogue;

  // 
  // protected methods
  //

  //
  // Device-only utility methods
  //

  /// Iterator for fetching tile fragments from A
  CUTLASS_DEVICE
  typename Mma::IteratorA init_iterator_A(
    TileWorkDesc &tile_work
    //,GemmUniversalMode mode
    )
  {
    // The input A matrix
    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);

    // Update input pointers based on batched/array mode
    // if (mode == GemmUniversalMode::kBatched) {
    //   ptr_A += tile_work.tiled_coord.k() * params.batch_stride_A;
    // }
    // if (mode == GemmUniversalMode::kArray) {
    //   ptr_A = static_cast<ElementA * const *>(params.ptr_A)[tile_work.tiled_coord.k()];
    // }

    ptr_A = static_cast<ElementA * const *>(params.ptr_A)[tile_work.tiled_coord.k()];

    int m_begin = tile_work.tiled_coord.m() * Mma::Shape::kM;
    int m_end = params.block_mapping.problem_size.m();
    return Mma::IteratorA(
        params.params_A,
        ptr_A,
        { m_end, tile_work.k_end },
        threadIdx.x,
        { m_begin, tile_work.k_begin });

  }

  /// Iterator for fetching tile fragments from B
  CUTLASS_DEVICE
  typename Mma::IteratorB init_iterator_B(
    TileWorkDesc &tile_work
    //,GemmUniversalMode mode
    )
  {
    // The input B matrix
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    // Update input pointers based on batched/array mode
    // if (mode == GemmUniversalMode::kBatched) {
    //   ptr_B += tile_work.tiled_coord.k() * params.batch_stride_B;
    // }
    // if (mode == GemmUniversalMode::kArray) {
    //   ptr_B = static_cast<ElementB * const *>(params.ptr_B)[tile_work.tiled_coord.k()];
    // }

    ptr_B = static_cast<ElementB * const *>(params.ptr_B)[tile_work.tiled_coord.k()];

    int n_begin = tile_work.tiled_coord.n() * Mma::Shape::kN;
    int n_end = params.block_mapping.problem_size.n();
    return Mma::IteratorB(
        params.params_B,
        ptr_B,
        { tile_work.k_end, n_end },
        threadIdx.x,
        { tile_work.k_begin, n_begin });
  }

  CUTLASS_DEVICE
  void init_dp_tile_work(
      TileWorkDesc &tile_work,
      int tile_idx)
  {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    tile_work.iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = params.block_mapping.iters_per_tile();

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_begin = 0;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_end = params.block_mapping.problem_size.k();

    // The location of this tile (in threadblock-tile coordinates) in the output matrix
    tile_work.tiled_coord = params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }

  CUTLASS_DEVICE
  void init_sk_tile_work(
      TileWorkDesc &tile_work,
      int tile_idx,
      int block_iter_begin,
      int block_iter_end)
  {
    // The linear tile index
    tile_work.tile_idx = tile_idx;

    // The first global-scoped MAC-iteration for this tile
    int tile_iter_begin = tile_idx * params.block_mapping.iters_per_tile();

    // The first global-scoped MAC-iteration this threadblock will perform for this tile
    tile_work.iter_begin = max(block_iter_begin, tile_iter_begin);

    // The first tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_begin = tile_work.iter_begin - tile_iter_begin;

    // The last (one past) tile-scoped MAC-iteration this threadblock will perform for this tile
    int k_iter_end = block_iter_end - tile_iter_begin;

    // The number of MAC-iterations this threadblock will perform for this tile
    tile_work.k_iters_remaining = k_iter_end - k_iter_begin;

    // The starting index in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_begin = k_iter_begin * Mma::Shape::kK;

    // The ending index (one-past) in the k-domain for MAC-iterations this threadblock will perform for this tile
    tile_work.k_end = min(
        params.block_mapping.problem_size.k(),            // extent of k domain
        (k_iter_end * Mma::Shape::kK));                   // extent of the threadblock's global iteration assignment

    // The location of this tile (in threadblock-tile coordinates) in the output matrix
    tile_work.tiled_coord = params.block_mapping.get_tile_offset(tile_work.tile_idx);
  }

  /// Share accumulators with peers
  CUTLASS_DEVICE
  void share_accumulators(
    AccumulatorTile const &accumulator_tile,
    int block_idx,
    int first_block_idx)
  {
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    int accum_tile_offset = first_block_idx * kThreadCount;

    if (block_idx == first_block_idx)
    {
      // First peer initializes the workspace partials
      BlockStripedReduceT::store(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }
    else
    {
      // Subsequent peers atomically accumulate into the workspace partials
      if (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic)
      {
        // Non-deterministic reduction order: wait for the first peer to have initialized the partials before we add to them
        Barrier::wait_lt(params.barrier_workspace, thread_idx, first_block_idx, 1);
      }
      else
      {
        // Turnstile reduction order: wait until the previous peer has written
        int wait_count = block_idx - first_block_idx;
        Barrier::wait_eq(params.barrier_workspace, thread_idx, first_block_idx, wait_count);
      }

      // Perform reduction in workspace
      BlockStripedReduceT::reduce(accum_tile_workspace + accum_tile_offset, accumulator_tile, thread_idx);
    }

    // Signal our arrival
    Barrier::arrive_inc(params.barrier_workspace, thread_idx, first_block_idx);
  }

  /// Acquire accumulators from peers
  CUTLASS_DEVICE
  void acquire_accumulators(
    AccumulatorTile &accumulator_tile,
    int block_idx,
    int first_block_idx)
  {
    AccumulatorTile *accum_tile_workspace = reinterpret_cast<AccumulatorTile *>(params.partials_workspace);

    // Wait for arrival
    int num_carry_in = block_idx - first_block_idx;
    Barrier::wait_eq_reset(params.barrier_workspace, thread_idx, first_block_idx, num_carry_in);

    // Load and add peer-partials accumulator tile to local accumulator tile
    int accum_tile_offset = first_block_idx * kThreadCount;
    BlockStripedReduceT::load_add(accumulator_tile, accum_tile_workspace + accum_tile_offset, thread_idx);
  }

  /// Perform epilogue computations and output
  CUTLASS_DEVICE
  void do_epilogue(
    TileWorkDesc &tile_work,
    AccumulatorTile &accumulator_tile)
  {
    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    // Update pointers for batched/array mode(s)
    if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C += tile_work.tiled_coord.k() * params.batch_stride_C;
      ptr_D += tile_work.tiled_coord.k() * params.batch_stride_D;
    }
    if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[tile_work.tiled_coord.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[tile_work.tiled_coord.k()];
    }

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
      tile_work.tiled_coord.m() * Mma::Shape::kM,
      tile_work.tiled_coord.n() * Mma::Shape::kN
    );

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Execute the epilogue operator to update the destination tensor.
    epilogue(
        EpilogueOutputOp(params.output_op),
        iterator_D,
        accumulator_tile,
        iterator_C);
  }

  CUTLASS_DEVICE
  void separate_reduction(int reduce_idx)
  {
    int peer_idx_begin, peer_idx_last, reduce_tile_idx, reduce_fragment_idx;

    // Reduce by sk-tile (every tile contributed to by one or more blocks)
    reduce_tile_idx = reduce_idx / Epilogue::kAccumulatorFragments;
    reduce_fragment_idx = reduce_idx % Epilogue::kAccumulatorFragments;

    int iter_tile_first = reduce_tile_idx * params.block_mapping.iters_per_tile();
    int iter_tile_last = iter_tile_first + params.block_mapping.iters_per_tile() - 1;

    peer_idx_begin = params.block_mapping.get_sk_block_idx(iter_tile_first);
    peer_idx_last = params.block_mapping.get_sk_block_idx(iter_tile_last);

    // Wait for peers to complete
    int peer_idx_end = peer_idx_last + 1;
    int num_peers = peer_idx_end - peer_idx_begin;
    Barrier::wait_eq_reset(
        params.barrier_workspace,
        thread_idx,
        (reduce_tile_idx * Epilogue::kAccumulatorFragments) + reduce_fragment_idx,
        num_peers);

    /// The location of this tile (in threadblock-tile coordinates) in the output matrix
    GemmCoord tiled_coord = params.block_mapping.get_tile_offset(reduce_tile_idx);

    // Location of this tile in item-coords
    MatrixCoord threadblock_item_begin(
      tiled_coord.m() * Mma::Shape::kM,
      tiled_coord.n() * Mma::Shape::kN
    );

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.block_mapping.problem_size.mn(),
        thread_idx,
        threadblock_item_begin);

    // Execute the epilogue operator to update the destination tensor.
    epilogue.reduce(
        peer_idx_begin,
        peer_idx_end,
        reduce_fragment_idx,
        params.partials_workspace,
        EpilogueOutputOp(params.output_op),
        iterator_D,
        iterator_C);
  }

  CUTLASS_DEVICE
  void process_tile(
    TileWorkDesc tile_work,
    int block_idx,
    int dp_start_block_idx,
    int block_iter_begin)
  {
    // Initialize input iterators
    typename Mma::IteratorA iterator_A = init_iterator_A(tile_work, params.mode);
    typename Mma::IteratorB iterator_B = init_iterator_B(tile_work, params.mode);

    // Initialize accumulators
    AccumulatorTile accumulator_tile;
    accumulator_tile.clear();

    // Initialize MMA abstraction
    Mma mma(
      shared_storage.main_loop,
      thread_idx,
      warp_idx,
      lane_idx);

    // Perform this tile's range of multiply-accumulate (MAC) iterations
    mma(tile_work.k_iters_remaining, accumulator_tile, iterator_A, iterator_B, accumulator_tile);

    if ((ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kAtomic) ||
        (params.block_mapping.reduction_blocks == 0) ||
        (block_idx >= dp_start_block_idx))
    {
      //
      // Cooperative SK peer reduction or DP block
      //

      int first_block_idx = params.block_mapping.get_first_block_idx(tile_work.tile_idx, block_idx);

      if (!tile_work.tile_finished(params)) {
        // Non "finishing" SK blocks must share their partial accumulator sums through global scratch workspace
        share_accumulators(accumulator_tile, block_idx, first_block_idx);
      }
      else
      {
        // DP blocks and "finishing" SK blocks must perform epilogue operations and write the output tile
        if (!tile_work.tile_started())
        {
          // A "finishing" SK block must first aggregate its accumulator partial sums with those shared by peer threadblocks
          acquire_accumulators(accumulator_tile, block_idx, first_block_idx);
        }

        do_epilogue(tile_work, accumulator_tile);
      }
    }
    else
    {
      //
      // Separate peer reduction
      //

      // Share accumulator partial sums with peer threadblock(s) through scratch workspace
      epilogue.share(block_idx, params.partials_workspace, accumulator_tile, tile_work.tile_started());

      // Signal arrival
      Barrier::arrive_range_inc(
        params.barrier_workspace,
        thread_idx,
        tile_work.tile_idx * Epilogue::kAccumulatorFragments,
        Epilogue::kAccumulatorFragments);
    }
  }

  /// Executes one implicit GEMM
  CUTLASS_DEVICE
  void implicit_gemm()
  {
    // Initialize block's iteration range
    int tile_idx = 0;
    int block_iter_begin = 0;
    int block_iters_remaining = 0;

    int block_idx = params.block_mapping.get_block_idx();

    int sk_padding_start_block_idx =  params.block_mapping.sk_regions() * params.block_mapping.sk_blocks_per_region();
    int dp_start_block_idx = params.block_mapping.sk_waves * params.block_mapping.avail_sms;
    int reduce_start_block_idx = dp_start_block_idx + params.block_mapping.dp_blocks;
    int grid_padding_start_block_idx = reduce_start_block_idx + params.block_mapping.reduction_blocks;

    // Initialize tile work descriptor
    TileWorkDesc tile_work;

    bool dp_block = (block_idx >= dp_start_block_idx) && (block_idx < reduce_start_block_idx);
    bool sk_block = (block_idx < sk_padding_start_block_idx);
    bool reduce_block = (block_idx >= reduce_start_block_idx) &&
            (block_idx < grid_padding_start_block_idx) &&
            (ThreadblockSwizzle::kReductionStrategy == ThreadblockSwizzle::kMixed);

    if (dp_block)
    {
      // This is a DP block
      int dp_block_idx = block_idx - dp_start_block_idx;
      int first_dp_tile = (params.block_mapping.cohort_raster) ? 0 : params.block_mapping.sk_tiles;

      // Blocks in first DP wave get configured number of tiles
      tile_idx = first_dp_tile + dp_block_idx;
      int tile_allottment = params.block_mapping.dp_first_wave_tiles;

      // Blocks in subsequent DP waves get 1 tile
      if (dp_block_idx >= params.block_mapping.avail_sms) {
          tile_allottment = 1;
          tile_idx += (params.block_mapping.dp_first_wave_tiles - 1) * params.block_mapping.avail_sms;
      }

      block_iters_remaining = params.block_mapping.iters_per_tile() * tile_allottment;

      init_dp_tile_work(tile_work, tile_idx);

      // DP blocks exit if out of bounds or overlap an SK tile (only possible during cohort rasterization, where dp_first_wave_tiles must be 1)
      if ((tile_idx < params.block_mapping.sk_tiles) ||
          (tile_work.tiled_coord.m() >= params.block_mapping.tiled_shape().m()) ||
          (tile_work.tiled_coord.n() >= params.block_mapping.tiled_shape().n()))
      {
        return;
      }
    }
    else if (sk_block)
    {
      // This is a SK block
      int block_iter_end;
      params.block_mapping.get_iter_extents(block_idx, block_iter_begin, block_iter_end);
      block_iters_remaining = block_iter_end - block_iter_begin;

      tile_idx = params.block_mapping.get_sk_tile_idx(block_iter_end - 1);
      init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
    }
    else
    {
      if (reduce_block)
      {
        // This is a reduction threadblock
        int reduce_block_idx = block_idx - reduce_start_block_idx;
        separate_reduction(reduce_block_idx);
      }

      return;
    }

    // Iteration-processing loop body
    CUTLASS_PRAGMA_NO_UNROLL
    while (true)
    {
      // Perform this block's share of work for this tile
      process_tile(
        tile_work,
        block_idx,
        dp_start_block_idx,
        block_iter_begin);

      block_iters_remaining -= tile_work.k_iters_remaining;

      if (block_iters_remaining == 0)
      {
        break;
      }

      // Continue to next tile
      __syncthreads();

      if (block_idx >= dp_start_block_idx)
      {
        // DP block consume their tiles at stride
        tile_idx += params.block_mapping.avail_sms;
        init_dp_tile_work(tile_work, tile_idx);
      }
      else
      {
        // SK blocks consume their tiles in backwards order
        tile_idx--;
        init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
      }
    }

  }

public:

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolutionStreamK() { } 

  CUTLASS_DEVICE
  void operator()(){
    implicit_gemm();
  }

#if 0
  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        params.block_mapping.get_tile_offset();

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;
    if (kGroupMode != GroupMode::kNone) {
      if (kGroupMode != GroupMode::kDepthwise) {
        int k_per_group = params.problem_size.K / params.problem_size.groups;
        int group_idx = threadblock_tile_idx.n() * Mma::Shape::kN / k_per_group;
        int channels_per_group = params.problem_size.C / params.problem_size.groups;
        iterator_A_column_offset += group_idx * channels_per_group;
      } else {
        iterator_A_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
      }
    } 

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        iterator_A_column_offset
      )
    );
    
    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_B,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      )
    );

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, params.gemm_k_iterations_per_channel);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    // Construct the semaphore.
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);
    
    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      params.ptr_D,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );
    
    // Tile iterator reading from source accumulator tensor
    typename Epilogue::OutputTileIterator iterator_C(
      params.iterator_C,
      params.ptr_C,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );


    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_idx.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_idx.k());

    }
    // Each split-k-slice writes to a unique tensor location
    else if (params.split_k_mode == SplitKMode::kParallel) {
      iterator_D.add_pointer_offset(threadblock_tile_idx.k() * 
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size));
    }

    // Run efficient epilogue
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  
    //
    // Release the semaphore
    //

    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) { 

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_idx.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_idx.k() + 1;
      }
      
      semaphore.release(lock);
    }
  } 
#endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
