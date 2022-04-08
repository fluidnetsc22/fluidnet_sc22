/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
//#include </softs/src/cuda-10.0/samples/common/inc/helper_functions.h>  // helper for shared functions common to CUDA Samples
//#include </softs/src/cuda-10.0/samples/common/inc/helper_cuda.h>       // helper function CUDA error checking and initialization
#include <helper_functions.h>
#include <helper_cuda.h>

#include "CG.h"
#include "fluids_init.h"
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <torch/extension.h>

namespace fluid {

typedef at::Tensor T;

const char *sSDKname     = "conjugateGradient";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (float)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}


//int Conjugate_Gradient
void Conjugate_Gradient
(
 T flags,
 T div_vec,
 T A_val,
 T I_A,
 T J_A,
 const float tol,
 const int max_iter,
 int M,
 int N,
 int nz,
 T p,
 float &r1
){
    int *I = NULL, *J = NULL;
    float *val = NULL;
    float *x;

    float *rhs;
    float a, b, na, r0;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;

    T residual = zeros_like(flags);
    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    //int devID = findCudaDevice(argc, (const char **)argv);


    // Previous declaration
    I = I_A.data<int>();
    J = J_A.data<int>();
    val = A_val.data<float>();
    rhs = div_vec.data<float>();
    x = p.data<float>();

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseSpMatDescr_t matA = NULL;
    checkCudaErrors(cusparseCreateCsr(
        &matA, N, N, nz, d_row, d_col, d_val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_32F));
    cusparseDnVecDescr_t vecp = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
    cusparseDnVecDescr_t vecAx = NULL;
    checkCudaErrors(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_32F));

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);


    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    /* Allocate workspace for cuSPARSE */
    size_t bufferSize = 0;
    checkCudaErrors(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    void *buffer = NULL;
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));

    /* Begin CG */
    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffer));

    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);





    k = 1;


    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }


         checkCudaErrors(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
            vecp, &beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &buffer));
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaDeviceSynchronize();
        k++;
    }

    //printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    float rsum, diff, err = 0.0;


    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    if (matA       ) { checkCudaErrors(cusparseDestroySpMat(matA)); }
    if (vecx       ) { checkCudaErrors(cusparseDestroyDnVec(vecx)); }
    if (vecAx      ) { checkCudaErrors(cusparseDestroyDnVec(vecAx)); }
    if (vecp       ) { checkCudaErrors(cusparseDestroyDnVec(vecp)); }

    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

}



}
