// Uncomment this line to choose between FastRand or Bucket 
#define USE_FASTRAND
// #define USE_BUCKET

namespace dgl {

using namespace cuda;

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
    float acc = 0.0; 
    int nnz = hb - lb;
    float norm;

    if (nnz < s)
      norm = float(nnz);
    else
      norm = float(s);

    for (int ss = threadIdx.x; ss < s && (lb+ss) < hb; ss+=blockDim.x) {
      sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        offset = sh[(sm_offset+kk)] + cid;
        // acc1 = sum_reduce(acc1, B[offset]);
        acc += B[offset];
      }
      offset = rid*k + cid;
      C[offset] = acc/norm;
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
    float acc = 0.0;
    int nnz = hb - lb;
    float norm;

    if (nnz < s)
      norm = float(nnz);
    else
      norm = float(s);

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
        acc += B[offset];
      }
      offset = rid*k + cid;
      C[offset] = acc/norm;
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
    float acc = 0.0; //sum_init();

    for (int ss = threadIdx.x; ss < s && (lb+ss) < hb; ss+=blockDim.x) {
      sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
      sh_data[(sm_offset + ss)] = A_data[(lb + ss)];
    }
    __syncthreads();

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        offset = sh_indices[(sm_offset+kk)] + cid;
        acc += sh_data[sm_offset+kk] * B[offset];
      }
      offset = rid*k + cid;
      C[offset] = acc;
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
    float acc = 0.0; //sum_init();

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
        // acc1 = sum_reduce(acc1, sh_data[sm_offset+kk]*B[offset]);
        acc += sh_data[sm_offset+kk]*B[offset];
      }
      offset = rid*k + cid;
      C[offset] = acc;
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
} // namespace dgl
