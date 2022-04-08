#pragma once

#include <sstream>
#include <vector>

#include "torch/extension.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>  // helper function CUDA error checking and initialization

#include "torch/extension.h"


namespace fluid {

typedef at::Tensor T;

void Conjugate_Gradient
(
 T flags,
 T div_vec,
 T A_val,
 T I_A,
 T J_A,
 const float p_tol,
 const int max_iter,
 int h,
 int w,
 int nnz,
 T p,
 float &r1
);


}
