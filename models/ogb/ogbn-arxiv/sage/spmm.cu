/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SPMM C APIs and definitions.
 */
// #define KERNEL_TIME
// #define KERNEL_INFO
// #define CALL_FUNC
#define USE_CACHE_SAMPLE

#include <dgl/array.h>
#include "./spmm.cuh"
#include "./functor.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./csspmm.cuh"

namespace dgl {

using namespace cuda;

namespace aten {
namespace {

/*! \brief Fill the vector started from ptr of size length with val */
template <typename DType>
void _Fill(DType* ptr, size_t length, DType val) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = FindNumThreads(length);
  int nb = (length + nt - 1) / nt;  // on x-axis, no need to worry about upperbound.
  CUDA_KERNEL_CALL(cuda::_FillKernel, nb, nt, 0, thr_entry->stream, ptr, length, val);
}

}  // namespace

namespace cusparse {

#if CUDART_VERSION < 11000
template <typename DType>
cusparseStatus_t Xcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA,
    const DType* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA,
    const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA,
    const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}
#endif

template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

/*! Cusparse implementation of SpMM on Csr format. */
template <typename IdType, typename DType>
void CusparseCsrmm2(
    const DLContext& ctx,
    const CSRMatrix& csr,
    const DType* B_data, const DType* A_data,
    DType* C_data,
    int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix for node
  // feature tensor. However, since cusparse only supports column-major, while our tensor
  // is stored in row-major, the actual computation is:
  // C = trans(A x trans(B)).
  // Currently, we use cublasXgeam to implement transposition and allocate intermediate
  // workspace memory for this.
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
#ifdef KERNEL_INFO
  static int m_acc = 0;
  static int nnz_acc = 0;
  m_acc += m;
  nnz_acc += nnz;
  float nnz_row = float(nnz)/float(m);
  float avg_nnz_row = float(nnz_acc)/float(m_acc);

  std::cout << __LINE__ << ": m, n, k, nnz = " << m << ", " << n << ", " << k << ", " << nnz
            << "; nnz/row = " << nnz_row << ", avg nnz/row = " << avg_nnz_row << std::endl;
#endif
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, thr_entry->stream));
  // allocate matrix for temporary transposed output
  DType* trans_out = static_cast<DType*>(device->AllocWorkspace(ctx, m * n * sizeof(DType)));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr = static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
#if CUDART_VERSION >= 11000
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  constexpr auto cuda_dtype = std::is_same<DType, float>::value ? CUDA_R_32F: CUDA_R_64F;

  if (sizeof(IdType) == 4){
    CUSPARSE_CALL(cusparseCreateCsr(&matA,
      m, k, nnz,
      static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data),
      const_cast<DType*>(valptr? valptr : A_data),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
  } else if(sizeof(IdType) == 8) {
    CUSPARSE_CALL(cusparseCreateCsr(&matA,
      m, k, nnz,
      static_cast<int64_t*>(csr.indptr->data),
      static_cast<int64_t*>(csr.indices->data),
      const_cast<DType*>(valptr? valptr : A_data),
      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
  } else {
    LOG(FATAL) << "Unsupported IdType";
  }
  /*
  CUSPARSE_CALL(cusparseCreateCsr(&matA,
      m, k, nnz,
      // static_cast<int32_t*>(csr.indptr->data),
      // static_cast<int32_t*>(csr.indices->data),
      static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr? valptr : A_data),
      // CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      cusparse_idx_type, cusparse_idx_type,
      CUSPARSE_INDEX_BASE_ZERO, cuda_dtype));
  */
  CUSPARSE_CALL(cusparseCreateDnMat(&matB,
      n, k, n,
      const_cast<DType*>(B_data), cuda_dtype, CUSPARSE_ORDER_COL));
  CUSPARSE_CALL(cusparseCreateDnMat(&matC,
      m, n, m,
      trans_out, cuda_dtype, CUSPARSE_ORDER_COL));

  auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = CUSPARSE_OPERATION_TRANSPOSE;
  size_t workspace_size;
  CUSPARSE_CALL(cusparseSpMM_bufferSize(
      thr_entry->cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      // cuda_dtype, CUSPARSE_CSRMM_ALG1,
      cuda_dtype, CUSPARSE_SPMM_CSR_ALG2,
      &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  CUSPARSE_CALL(cusparseSpMM(
      thr_entry->cusparse_handle, transA, transB,
      &alpha, matA, matB, &beta, matC,
      // cuda_dtype, CUSPARSE_CSRMM_ALG1,
      cuda_dtype, CUSPARSE_SPMM_CSR_ALG2,
      workspace));
  device->FreeWorkspace(ctx, workspace);

  CUSPARSE_CALL(cusparseDestroySpMat(matA));
  CUSPARSE_CALL(cusparseDestroyDnMat(matB));
  CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#else
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE,
      m, n, k, nnz, &alpha,
      descr, (valptr)? valptr : A_data,
      static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data),
      B_data, n, &beta, trans_out, m));
  CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
#endif
  if (valptr)
    device->FreeWorkspace(ctx, valptr);
  // transpose the output matrix
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, thr_entry->stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n, m,
      &alpha, trans_out, m,
      &beta, nullptr, n,
      C_data, n));
  device->FreeWorkspace(ctx, trans_out);
}
/*
__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0.0;
}

// Idtype = int32_t
template<typename IdType>
__global__ void CacheSampleSpMM_Bucket(
  const int m, const int k, const int s,
  const IdType* A_indptr, const IdType* A_indices,
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
    float acc1 = sum_init();
    int nnz = hb - lb;
    float norm = 0.0;

    if (nnz < s)
      norm = 1/float(nnz);
    else
      norm = 1/float(s);

    for (int ss = threadIdx.x; ss < s && (lb+ss) < hb; ss+=blockDim.x) {
      sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        offset = sh[(sm_offset+kk)] + cid;
        acc1 = sum_reduce(acc1, B[offset]);
      }
      offset = rid*k + cid;
      C[offset] = acc1*norm;
    }
  }
}

// IdType = int32_t
template<typename IdType>
__global__ void CacheSampleSpMM_FastRand(
  const int m, const int k, const int s,
  const IdType* A_indptr, const IdType* A_indices,
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
    float acc1 = sum_init();
    int nnz = hb - lb;
    float norm = 0.0;

    if (nnz < s)
      norm = 1/float(nnz);
    else
      norm = 1/float(s);

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
        acc1 = sum_reduce(acc1, B[offset]);
      }
      offset = rid*k + cid;
      C[offset] = acc1*norm;
    }
  }
}

// IdType = int32_t
__global__ void CacheSampleSpMM_Mul_Bucket(
  const int m, const int k, const int s,
  const int32_t* A_indptr, const int32_t* A_indices,
  const float* A_data,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int *sh_indices = sh;
  float *sh_data = (float*)&sh_indices[(s*blockDim.y)];
  int sm_offset = threadIdx.y*s;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    float acc1 = sum_init();

    for (int ss = threadIdx.x; ss < s && (lb+ss) < hb; ss+=blockDim.x) {
      sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
      sh_data[(sm_offset + ss)] = A_data[(lb + ss)];
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        offset = sh_indices[(sm_offset+kk)] + cid;
        acc1 = sum_reduce(acc1, sh_data[sm_offset+kk]*B[offset]);
      }
      offset = rid*k + cid;
      C[offset] = acc1;
    }
  }
}

// IdType = int32_t
__global__ void CacheSampleSpMM_Mul_FastRand(
  const int m, const int k, const int s,
  const int32_t* A_indptr, const int32_t* A_indices,
  const float* A_data,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int *sh_indices = sh;
  float *sh_data = (float*)&sh_indices[(s*blockDim.y)];
  int sm_offset = threadIdx.y*s;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int nnz = hb - lb;
    int offset;
    float acc1 = sum_init();

    if (nnz < s) {
      for (int ss = threadIdx.x; (lb+ss) < hb; ss+=blockDim.x) {
        sh_indices[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
        sh_data[(sm_offset + ss)] = A_data[(lb + ss)];
      }
    }
    else {
      for (int ss = threadIdx.x; ss < s; ss+=blockDim.x) {
        offset = lb + ((ss*577) % nnz);
        sh_indices[(sm_offset + ss)] = A_indices[offset]*k;
        sh_data[(sm_offset + ss)] = A_data[offset];
      }
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        offset = sh_indices[(sm_offset+kk)] + cid;
        acc1 = sum_reduce(acc1, sh_data[sm_offset+kk]*B[offset]);
      }
      offset = rid*k + cid;
      C[offset] = acc1;
    }
  }
}

template <typename IdType, typename DType>
void XCacheSampleCsrmm(
  int m, int n, int s,
  const IdType* A_indptr,
  const IdType* A_indices,
  const DType* B_data, DType* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  LOG(FATAL) << "Not supported yet";
}

template <>
void XCacheSampleCsrmm<int32_t, float>(
  int m, int n, int s,
  const int32_t* A_indptr,
  const int32_t* A_indices,
  const float* B_data, float* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#ifdef USE_FASTRAND
  CUDA_KERNEL_CALL(CacheSampleSpMM_FastRand<int32_t>, 
    grid, block, shmem, 
    thr_entry->stream, 
    m, n, s,
    A_indptr,
    A_indices,
    B_data, C_data);
#endif
#ifdef USE_BUCKET
  CUDA_KERNEL_CALL(CacheSampleSpMM_Bucket<int32_t>, 
    grid, block, shmem, thr_entry->stream, 
    m, n, s,
    A_indptr,
    A_indices,
    B_data, C_data);
#endif
  cudaStreamSynchronize(thr_entry->stream);
}

template <>
void XCacheSampleCsrmm<int64_t, float>(
  int m, int n, int s,
  const int64_t* A_indptr,
  const int64_t* A_indices,
  const float* B_data, float* C_data,
  dim3 grid, dim3 block,
  int shmem) {

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#ifdef USE_FASTRAND
  CUDA_KERNEL_CALL(CacheSampleSpMM_FastRand<int64_t>, 
    grid, block, shmem,
    thr_entry->stream, 
    m, n, s,
    A_indptr,
    A_indices,
    B_data, C_data);
#endif
#ifdef USE_BUCKET
  CUDA_KERNEL_CALL(CacheSampleSpMM_Bucket<int64_t>, 
    grid, block, shmem, thr_entry->stream, 
    m, n, s,
    A_indptr,
    A_indices,
    B_data, C_data);
#endif
  cudaStreamSynchronize(thr_entry->stream);
}

template <typename DType>
void XCacheSampleCsrmmMul(
  int m, int n, int s,
  const int32_t* A_indptr,
  const int32_t* A_indices,
  const DType* A_data,
  const DType* B_data, DType* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  LOG(FATAL) << "Not supported yet";
}

template <>
void XCacheSampleCsrmmMul<float>(
  int m, int n, int s,
  const int32_t* A_indptr,
  const int32_t* A_indices,
  const float* A_data,
  const float* B_data, float* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#ifdef USE_FASTRAND
  CUDA_KERNEL_CALL(CacheSampleSpMM_Mul_FastRand, 
    grid, block, shmem*2, thr_entry->stream, 
    m, n, s,
    A_indptr, A_indices, A_data,
    B_data, C_data);
#endif
#ifdef USE_BUCKET
  CUDA_KERNEL_CALL(CacheSampleSpMM_Mul_Bucket, 
    grid, block, shmem*2, thr_entry->stream, 
    m, n, s,
    A_indptr, A_indices, A_data,
    B_data, C_data);
#endif
  cudaStreamSynchronize(thr_entry->stream);
}
*/
void printKernelInfo(char name[], dim3 grid, dim3 blk, int shmem, 
        int m, int n, int s) {
  std::cout << name << "<<<(" << grid.x << ", " << grid.y << ", " << grid.z
            << "), (" << blk.x << ", " << blk.y << ", " << blk.z << "), "
            << shmem << ")>>>";
  std::cout << " with (m, n, s) = " << "(" << m << ", " << n 
      << ", " << s << ")" << std::endl;
}

template <typename IdType, typename DType>
void CacheSampleCsrmm(
    const DLContext& ctx,
    const aten::CSRMatrix& csr,
    const DType* B_data, //const DType* A_data,
    DType* C_data,
    int x_length, 
    const int S) {
  const int M = csr.num_rows;
  const int N = x_length;

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

#ifdef KERNEL_INFO
  char kernel_name[] = "CacheSampleCsrmm()";
  printKernelInfo(kernel_name, grid, block, shmem, M, N, S);
#endif

  XCacheSampleCsrmm<IdType, DType>(
    M, N, S,
    static_cast<IdType*>(csr.indptr->data),
    static_cast<IdType*>(csr.indices->data),
    B_data, C_data,
    grid, block, shmem);
}

template <typename DType>
void CacheSampleCsrmmMul(
    const DLContext& ctx,
    const aten::CSRMatrix& csr,
    const DType* B_data, const DType* A_data,
    DType* C_data,
    int x_length, 
    const int S) {
  const int M = csr.num_rows;
  const int N = x_length;

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

#ifdef KERNEL_INFO
  char kernel_name[] = "CacheSampleCsrmmMul()";
  printKernelInfo(kernel_name, grid, block, shmem, M, N, S);
#endif

  XCacheSampleCsrmmMul<DType>(
    M, N, S,
    static_cast<int32_t*>(csr.indptr->data),
    static_cast<int32_t*>(csr.indices->data),
    A_data, B_data, C_data,
    grid, block, shmem);
}
/*
template <typename IdType, typename DType>
void GeSpmmCsrmm(
  const DLContext& ctx,
  const aten::CSRMatrix& csr,
  const DType* B_data, DType* C_data,
  int x_length) {

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int m = csr.num_rows;
  const int n = x_length;
  typedef int32_t Idx;

  XTopoCsrmm<DType>(ctx,
    m, n, 
    static_cast<IdType*>(csr.indptr->data),
    static_cast<IdType*>(csr.indices->data),
    B_data, C_data
  );

  cudaStreamSynchronize(thr_entry->stream);
  CUDA_CALL(cudaGetLastError());
}
*/
}  // namespace cusparse

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef cuda::binary::Add<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "sub") {                                     \
      typedef cuda::binary::Sub<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef cuda::binary::Mul<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "div") {                                     \
      typedef cuda::binary::Div<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_lhs") {                                \
      typedef cuda::binary::CopyLhs<DType> Op;                      \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_rhs") {                                \
      typedef cuda::binary::CopyRhs<DType> Op;                      \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux,
             const int S) {
#ifdef CALL_FUNC
  LOG(INFO) << "calling SpMMCsr()";
#endif
#ifdef KERNEL_TIME
  cudaEvent_t cuda_start, cuda_stop;
  cudaEventCreate(&cuda_start);
  cudaEventCreate(&cuda_stop);
  cudaEventRecord(cuda_start);
#endif
  if (reduce == "sum") {
#ifdef CALL_FUNC
    LOG(INFO) << "reduce == sum";
#endif
#if CUDART_VERSION < 11000
    if (sizeof(IdType) == 4 && op == "copy_lhs") {
#else
    if (op == "copy_lhs") {
#endif
#ifdef CALL_FUNC
      LOG(INFO) << "op == copy_lhs";
#endif
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i)
        x_length *= ufeat->shape[i];
      // SWITCH between cusparse and cache_sample kernel
#ifdef USE_CACHE_SAMPLE
      cusparse::CacheSampleCsrmm<IdType, DType>(
          ufeat->ctx, csr,
          static_cast<DType*>(ufeat->data),
          static_cast<DType*>(out->data),
          x_length, S);
// #endif
/*
#ifdef USE_GE_SPMM
      GeSpmmCsrmm<IdType, DType>(
          ufeat->ctx, csr,
          static_cast<DType*>(ufeat->data),
          static_cast<DType*>(out->data),
          x_length);
#endif
*/
// #ifdef USE_CUSPARSE
#else
      cusparse::CusparseCsrmm2<IdType, DType>(
          ufeat->ctx, csr,
          static_cast<DType*>(ufeat->data),
          nullptr,
          static_cast<DType*>(out->data),
          x_length);
#endif
    } else if (sizeof(IdType) == 4 && op == "mul" && efeat.NumElements() == csr.indices->shape[0]) {
#ifdef CALL_FUNC
      LOG(INFO) << "op == mul";
#endif
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i)
        x_length *= ufeat->shape[i];
      if (!IsNullArray(csr.data))
        efeat = IndexSelect(efeat, csr.data);
#ifdef USE_CACHE_SAMPLE
      cusparse::CacheSampleCsrmmMul(
          ufeat->ctx, csr,
          static_cast<DType*>(ufeat->data),
          static_cast<DType*>(efeat->data),
          static_cast<DType*>(out->data),
          x_length, S);
#else
      cusparse::CusparseCsrmm2<DType>(
          ufeat->ctx, csr,
          static_cast<DType*>(ufeat->data),
          static_cast<DType*>(efeat->data),
          static_cast<DType*>(out->data),
          x_length);
#endif
    } else {
#ifdef CALL_FUNC
      LOG(INFO) << "op == " << op;
#endif
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, NullArray(), NullArray());
      });
    }
  } else if (reduce == "max") {
#ifdef CALL_FUNC
    LOG(INFO) << "reduce == max";
#endif
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Max<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
#ifdef CALL_FUNC
    LOG(INFO) << "reduce == min";
#endif
    SWITCH_OP(op, Op, {
      cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Min<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
#ifdef KERNEL_TIME
  cudaEventRecord(cuda_stop);
  cudaEventSynchronize(cuda_stop);

  float msecPerSpMM = 0;
  cudaEventElapsedTime(&msecPerSpMM, cuda_start, cuda_stop);
  double flopsPerSpMM = 2.0 * static_cast<double>(csr.data->shape[0]) *
                              static_cast<double>(ufeat->shape[1]);
  double gigaFlops = (flopsPerSpMM * 1.0e-9f) /
                     (msecPerSpMM / 1000.0f);

  std::cout << __LINE__ << ": SpMMCsr() cudaEventElapsed time (ms): " 
            << msecPerSpMM << std::endl
            << " nnz: " << csr.data->shape[0]
            << " K: " << ufeat->shape[1]
            << " GFlops: " << flopsPerSpMM * 1.0e-9f << std::endl
            << " Performance (GFlops/s): " << gigaFlops << std::endl;
#endif
}

/*!
 * \brief CUDA implementation of g-SpMM on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Sum<IdType, DType, true> > (
          bcast, coo, ufeat, efeat, out, NullArray(), NullArray());
    });
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Max<IdType, DType, true> > (
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  }  else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Min<IdType, DType, true> > (
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template void SpMMCsr<kDLGPU, int32_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux,
    const int);
template void SpMMCsr<kDLGPU, int64_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux,
    const int);
template void SpMMCsr<kDLGPU, int32_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux,
    const int);
template void SpMMCsr<kDLGPU, int64_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux,
    const int);

template void SpMMCoo<kDLGPU, int32_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, float>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int32_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, double>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

}  // namespace aten
}  // namespace dgl
