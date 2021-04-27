#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <sys/time.h>
#include <stdexcept>
#include <fstream>
#include <ctime>

#include <cuda_runtime.h>
#include "cusparse.h"

#include "./util/mmio.hpp"
#include "./util/util.hpp"

using namespace std;

// #define VALIDATE

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    CLEANUP("Exit.");   \
    exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkCuSparseError( a ) do { \
    if (CUSPARSE_STATUS_SUCCESS != (a)) { \
    fprintf(stderr, "CuSparse runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    CLEANUP("Exit.");   \
    exit(EXIT_FAILURE); \
    } \
} while (0)

#define CLEANUP(s)                      \
do {                                    \
    printf("%s\n", s);                  \
    if (A_data) free(A_data);   \
    if (A_indptr) free(A_indptr);   \
    if (A_indices) free(A_indices); \
    if (B)  free(B);    \
    if (C)  free(C);    \
    if (golden) free(golden);   \
    if (A_data_dev) cudaFree(A_data_dev);     \
    if (A_indptr_dev) cudaFree(A_indptr_dev);         \
    if (A_indices_dev) cudaFree(A_indices_dev);     \
    if (B_dev) cudaFree(B_dev); \
    if (C_dev) cudaFree(C_dev); \
    if (start)      cudaEventDestroy(start);    \
    if (stop)       cudaEventDestroy(stop);     \
    if (descr)      cusparseDestroyMatDescr(descr);     \
    if (cusp_handle)     cusparseDestroy(cusp_handle);    \
    fprintf(fpo, "\n"); \
    fclose(fpo);    \
    cudaDeviceReset();                  \
    fflush(stdout);                     \
} while (0)


__global__ void warmup(){}

__global__ void CacheSampleSpMM_NoSample(
  const int m, const int k, const int s,
  const int* A_indptr, const int* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int sm_offset = threadIdx.y*s;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    float acc1 = 0.0;
    int nnz = hb - lb;

    for (int ss = threadIdx.x; (lb+ss) < hb; ss+=blockDim.x) {
      sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; (lb+kk) < hb; kk++) {
        offset = sh[(sm_offset+kk)] + cid;
        acc1 += B[offset];
      }
      offset = rid*k + cid;
      C[offset] = acc1;
    }
  }
}

__global__ void CacheSampleSpMM_Bucket(
  const int m, const int k, const int s,
  const int* A_indptr, const int* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int sm_offset = threadIdx.y*s;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    float acc1 = 0.0;
    int nnz = hb - lb;

    for (int ss = threadIdx.x; ss < s && (lb+ss) < hb; ss+=blockDim.x) {
      sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        offset = sh[(sm_offset+kk)] + cid;
        acc1 += B[offset];
      }
      offset = rid*k + cid;
      C[offset] = acc1;
    }
  }
}

__global__ void CacheSampleSpMM_FastRand(
  const int m, const int k, const int s,
  const int* A_indptr, const int* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int sm_offset = threadIdx.y*s;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    float acc1 = 0.0; 
    int nnz = hb - lb;

    if (nnz < s) {
      for (int ss = threadIdx.x; (lb+ss) < hb; ss+=blockDim.x)
        sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    else {
      for (int ss = threadIdx.x; ss < s; ss+=blockDim.x) {
        offset = lb + ((ss*577) % nnz);
        sh[(sm_offset + ss)] = A_indices[offset]*k;
      }
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        offset = sh[(sm_offset+kk)] + cid;
        acc1 += B[offset];
      }
      offset = rid*k + cid;
      C[offset] = acc1;
    }
  }
}

void spmmWrapper(int method,
                 int A_nrows, int B_ncols, 
                 int *A_rowPtr, int *A_colInd, float *A_val, 
                 float *B, float *C, 
                 const int S) {
  const int M = A_nrows;
  const int N = B_ncols;

  int DIM_X;
  int DIM_Y;

  if (N <= 32) {
    DIM_X = 32;
    DIM_Y = 4;
  }
  else if (N <= 128) {
    DIM_X = N;
    DIM_Y = 512/DIM_X;
  }
  else {
    DIM_X = 128;
    DIM_Y = 4;
  }

  int tile_k = (N+DIM_X-1)/DIM_X;
  int n_block = (M+DIM_Y-1)/DIM_Y;

  dim3 grid  = dim3(n_block, tile_k, 1);
  dim3 block = dim3(DIM_X, DIM_Y, 1);
  int shmem = (S*DIM_Y*sizeof(int));

  switch(method) {
    case 0:
      CacheSampleSpMM_Bucket<<<grid, block, shmem, 0>>>(
        M, N, S, A_rowPtr, A_colInd, B, C);
      break;
    case 1:
      CacheSampleSpMM_FastRand<<<grid, block, shmem, 0>>>(
        M, N, S, A_rowPtr, A_colInd, B, C);
      break;
    case 2:
      CacheSampleSpMM_NoSample<<<grid, block, shmem, 0>>>(
        M, N, S, A_rowPtr, A_colInd, B, C);
      break;
  }
}

int main(int argc, char** argv) {
    int A_nrows, A_ncols, nnz, B_ncols, max_ncols;
    int DEV_ID;
    int S_value;

    assert(argc > 4);
    DEV_ID = atoi(argv[2]);
    max_ncols = atoi(argv[3]);
    S_value = atoi(argv[4]);

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;

    // Host allocate
    int* A_indptr = 0;
    int* A_indices = 0;
    float* A_data = 0;
    float* B = 0;
    float* C = 0;
    float* golden = 0;
    float* A_data_dev = 0;
    int* A_indices_dev = 0;
    int* A_indptr_dev = 0;
    float* B_dev = 0;
    float* C_dev = 0;
    // float* C_tran_dev = 0;
    float rt, rt2;
    float avg_ms;
    float gflops;
    float one=1, zero=0;
    double tot_gflop;
    

    cudaEvent_t start, stop;
    cusparseHandle_t cusp_handle=0;
    // cublasHandle_t cubl_handle=0;
    cusparseMatDescr_t descr=0;
    cusparseStatus_t cusp_stat;
    // cublasStatus_t cubl_stat;

    FILE *fpo = fopen("csspmm_test.log", "a+");
    printf("reading file ...\n");
    readMtx<float>(argv[1], row_indices, col_indices, values, A_nrows, A_ncols, nnz);

    fprintf(fpo, "graph, %s, ", argv[1]);
    fprintf(fpo, "n_cols, %d, ", max_ncols);
    fprintf(fpo, "S_value, %d, ", S_value);

    A_data = (float *)malloc(nnz*sizeof(A_data[0]));
    A_indptr = (int *)malloc((A_nrows+1)*sizeof(A_indptr[0]));
    A_indices = (int *)malloc(nnz*sizeof(A_indices[0]));
    B = (float *)malloc((max_ncols*A_ncols)*sizeof(B[0]));
    
#ifdef VALIDATE
    C = (float *)malloc((A_nrows*max_ncols)*sizeof(C[0]));
    golden = (float *)malloc((A_nrows*max_ncols)*sizeof(golden[0]));
    if (!C || !golden) {
        CLEANUP("Host malloc failed\n");
        return 1;
    }
#endif
    if ( !A_data || !A_indices || !A_indptr || !B ) {
        CLEANUP("Host malloc failed\n");
        return 1;
    }

    /* format conversation COO -> CSR */
    for (int i=0; i<A_nrows+1; i++) {
        A_indptr[i] = 0;
    }
    for (int n=0; n<nnz; n++) {
        int row = row_indices[n];
        if (row>=A_nrows) fprintf(stderr, "out of bound row\n");
        A_indptr[row+1]++;
    }
    for (int n=1; n<A_nrows+1; n++) {
        A_indptr[n] += A_indptr[n-1];
    }
    for (int n=0; n<nnz; n++) {
        int ptr = A_indptr[row_indices[n]];
        if (col_indices[n]>A_ncols) fprintf(stderr, "out of bound column\n");
        A_indices[ptr] = col_indices[n];
        A_data[ptr] = 1;
        ptr++;
        A_indptr[row_indices[n]]=ptr;
    }
    for (int n=A_nrows-1; n>0; n--) {
        A_indptr[n] = A_indptr[n-1];
    }
    A_indptr[0] = 0; // COO->CSR finish
    
    printf("read file ok. N=%d nnz=%d\n", A_nrows, nnz);

    /* random assign */
    unsigned seed;
    seed = time(0);
    srand(seed);

    for (int i=0; i<max_ncols*A_ncols; i++) {
        B[i] = float(rand() %100 - 50)/100;
    }

    // allocate device memory
    cudaDeviceReset();
    cudaSetDevice(DEV_ID);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, DEV_ID );
    int max_threads_per_block = deviceProp.sharedMemPerBlock/(sizeof(int)+sizeof(float));
    if (max_threads_per_block > deviceProp.maxThreadsPerBlock) max_threads_per_block = deviceProp.maxThreadsPerBlock;
    // int max_threads_per_block = 1024;

    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5, cudaStat6;
    while (true) {
        cudaStat1 = cudaMalloc((void**)&A_indptr_dev, (A_nrows+1)*sizeof(A_indptr_dev[0]));
        cudaStat2 = cudaMalloc((void**)&A_indices_dev, nnz*sizeof(A_indices_dev[0]));
        cudaStat3 = cudaMalloc((void**)&A_data_dev, nnz*sizeof(A_data_dev[0]));
        cudaStat4 = cudaMalloc((void**)&B_dev, max_ncols*A_ncols*sizeof(B_dev[0]));
        cudaStat5 = cudaMalloc((void**)&C_dev, A_nrows*max_ncols*sizeof(C_dev[0]));
        // cudaStat6 = cudaMalloc((void**)&C_tran_dev, A_nrows*max_ncols*sizeof(C_tran_dev[0]));
        if ((cudaStat1 == cudaSuccess) && (cudaStat2 == cudaSuccess) && (cudaStat3 == cudaSuccess) &&
        (cudaStat4 == cudaSuccess) && (cudaStat5 == cudaSuccess)) {
        //  && (cudaStat6 == cudaSuccess)) {
            break;
        }
        cudaDeviceReset();
        cudaSetDevice(DEV_ID);
        max_ncols /= 2;
    }
    printf("max_ncols = %d\n", max_ncols);
    
    checkCudaError(cudaMemcpy(A_indptr_dev, A_indptr, (A_nrows+1)*sizeof(A_indptr_dev[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_indices_dev, A_indices, nnz*sizeof(A_indices_dev[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_data_dev, A_data, nnz*sizeof(A_data_dev[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(B_dev, B, max_ncols*A_ncols*sizeof(B_dev[0]), cudaMemcpyHostToDevice));
    
    cudaError_t cudaStat = cudaGetLastError();
    // device warm up
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) 
    {
        fprintf(stderr, "Warm-up failed: %s\t", cudaGetErrorString(cudaStat));
    }

    cusparseCreate(&cusp_handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    cusp_stat = cusparseScsrmm2(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, A_nrows, max_ncols, A_ncols, nnz, &one, descr, A_data_dev, A_indptr_dev, A_indices_dev, B_dev, max_ncols, &zero, C_dev, A_nrows);
    if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("csrmm2 failed");
        return 1;
    }

#define ITER 200

    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    
    for (int i=0; i<ITER; i++) {
        warmup<<<1,1>>>();
    }
    printf("running tests...\n");

    int tile_row = 8;

    B_ncols = max_ncols;

    tot_gflop = (double)nnz*2/1000000*B_ncols;

    cudaEventRecord(start, 0);
    for (int i=0; i<ITER; i++) {
        cusp_stat = cusparseScsrmm2(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, A_nrows, B_ncols, A_ncols, nnz, &one, descr, A_data_dev, A_indptr_dev, A_indices_dev, B_dev, B_ncols, &zero, C_dev, A_nrows);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&rt, start, stop);

    avg_ms = rt/ITER;
    gflops = tot_gflop/avg_ms;
    printf("cusparseScsrmm2, %f, %f, ", avg_ms, gflops); 
    fprintf(fpo, "cusparseScsrmm2, %f, %f, ", avg_ms, gflops); 
    ///*
    for (int method = 0; method < 2; method++) {
      cudaEventRecord(start, 0);
      for (int i=0; i<ITER; i++) {
          spmmWrapper(method, 
                      A_nrows, B_ncols, 
                      A_indptr_dev, A_indices_dev, A_data_dev, 
                      B_dev, C_dev, S_value);
      }
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&rt, start, stop);

      avg_ms = rt/ITER;
      gflops = tot_gflop/avg_ms;

      switch(method) {
        case 0:
          printf("csspmm_buc, %f, %f, ", avg_ms, gflops); 
          fprintf(fpo, "csspmm_buc, %f, %f, ", avg_ms, gflops); 
          break;
        case 1:
          printf("csspmm_fr, %f, %f, ", avg_ms, gflops); 
          fprintf(fpo, "csspmm_fr, %f, %f, ", avg_ms, gflops); 
          break;
      }
    }
    //*/

    /*
    cudaEventRecord(start, 0);
    for (int i=0; i<ITER; i++) {
        spmmWrapper(2, 
                    A_nrows, B_ncols, 
                    A_indptr_dev, A_indices_dev, A_data_dev, 
                    B_dev, C_dev, S_value);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&rt, start, stop);

    avg_ms = rt/ITER;
    gflops = tot_gflop/avg_ms;

    printf("csspmm_nosample, %f, %f, ", avg_ms, gflops); 
    fprintf(fpo, "csspmm_nosample, %f, %f, ", avg_ms, gflops); 
    */
    
    CLEANUP("");

    return 0;
}
