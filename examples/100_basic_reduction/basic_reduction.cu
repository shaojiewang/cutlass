// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision reduction kernel
//

// Defines cutlass::reduction::device::ReduceSplitK, the generic reduction computation template class.
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

// define a reduction function
template <class DTYPE>
cudaError_t CutlassReduceAll(int dim, DTYPE* src, DTYPE* dst)
{

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    DTYPE,
    128 / cutlass::sizeof_bits<DTYPE>::value,
    DTYPE,
    DTYPE
  >;

  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    DTYPE,
    DTYPE,
    EpilogueOutputOp::kCount
  >;

  using CutlassReduction = cutlass::reduction::device::ReduceSplitK<
    cutlass::reduction::kernel::ReduceSplitK<
      cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  cutlass::MatrixCoord problem_sizes = {128, 128};
  int partition = 2;
  

  CutlassReduction::Arguments args();
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_reduction example.
//
// usage:
//
//   100_basic_reduction <M> 
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain reduction dimensions and scalar values.
  //

  // reduction problem dimensions.
  int problem[1] = { 128 * 128 * 128 };

  for (int i = 0; i < argc && i < 1; ++i){
    std::stringstream ss(arg[i]);
    ss >> problem[i];
  }

  


  return 0;

}

