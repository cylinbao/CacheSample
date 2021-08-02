// Uncomment this line to choose between FastRand or Bucket 
#define USE_FASTRAND
// #define USE_BUCKET

namespace dgl {

using namespace cuda;

__global__ void CacheSampleCSRSpMM_v0(
  const int norm_bias, const int s, 
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
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
      norm = float(nnz + norm_bias);
    else
      norm = float(s + norm_bias);

    for (int ss = threadIdx.x; ss < s && (lb+ss) < hb; ss+=blockDim.x) {
      sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
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

__global__ void CacheSampleCSRSpMM_v1(
  const int norm_bias, const int s, 
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
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
      norm = float(nnz + norm_bias);
    else
      norm = float(s + norm_bias);

    if (nnz < s) {
      for (int ss = threadIdx.x; (lb+ss) < hb; ss += blockDim.x)
        sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    else {
      for (int ss = threadIdx.x; ss < s; ss += blockDim.x) {
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

__global__ void CacheSampleCSRSpMM_v2(
  const int norm_bias, const int s, const int p,
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
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
      norm = float(nnz + norm_bias);
    else
      norm = float(s + norm_bias);

    if (nnz < s) {
      for (int ss = threadIdx.x; (lb+ss) < hb; ss += blockDim.x)
        sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    else {
      for (int ss = threadIdx.x; ss < s; ss += blockDim.x) {
        offset = lb + ((ss*p) % nnz);
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

__global__ void CacheSampleCSRSpMM_v3(
  const int norm_bias, const int s, const int p, const unsigned int o,
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
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
      norm = float(nnz + norm_bias);
    else
      norm = float(s + norm_bias);

    if (nnz < s) {
      for (int ss = threadIdx.x; (lb+ss) < hb; ss += blockDim.x)
        sh[(sm_offset + ss)] = A_indices[(lb + ss)]*k;
    }
    else {
      for (int ss = threadIdx.x; ss < s; ss += blockDim.x) {
        int r = ((ss*ss) + o) % p;
        int x = (ss <= p / 2) ? r : p - r;
        offset = lb + (x % nnz);
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

template<typename IdType>
__global__ void SampleSpMM_FastRand(
  const int m, const int k, int s, int p,
  const IdType* A_indptr, const IdType* A_indices,
  // const int* A_indptr, const int* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  // int sm_offset = threadIdx.y*s;

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

    if (cid < k) {
      for (int kk = 0; kk < s && (lb+kk) < hb; kk++) {
        if (nnz < s) {
          offset = A_indices[(lb + kk)]*k + cid;
        }
        else {
          offset = lb + ((kk*p) % nnz);
          offset = A_indices[offset]*k + cid;
        }
        // offset = sh[(sm_offset+kk)] + cid;
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
  const std::string& kernel,
  const int norm_bias,
  const int s, 
  int m, int n, 
  const IdType* A_indptr,
  const IdType* A_indices,
  const DType* B_data, DType* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  LOG(FATAL) << "Not supported yet";
}

template <>
void XCacheSampleCsrmm<int32_t, float>(
  const std::string& kernel,
  const int norm_bias,
  const int s, 
  int m, int n, 
  const int32_t* A_indptr,
  const int32_t* A_indices,
  const float* B_data, float* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  int primes[10] = {577, 769, 983, 1193, 1429,
                    1619, 1871, 2089, 2339, 2579};
  int p = 21767;
  unsigned int offset;
  struct timespec tp;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  if (kernel.length() == 11) {
    clock_gettime(CLOCK_MONOTONIC, &tp);
    offset = (unsigned int) tp.tv_nsec;

    CUDA_KERNEL_CALL(CacheSampleCSRSpMM_v3, 
      grid, block, shmem, 
      thr_entry->stream, 
      norm_bias, s, p, offset,
      m, n, 
      A_indptr, A_indices,
      B_data, C_data);
  }
  else {
    if (kernel.substr(11, 2) == "V0") {
      CUDA_KERNEL_CALL(CacheSampleCSRSpMM_v0, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel.substr(11, 2) == "V1") {
      CUDA_KERNEL_CALL(CacheSampleCSRSpMM_v1, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel.substr(11, 2) == "V2") {
      srand(time(NULL));
      p = primes[rand() % 10];

      CUDA_KERNEL_CALL(CacheSampleCSRSpMM_v2, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s, p,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel.substr(11, 2) == "V3") {
      clock_gettime(CLOCK_MONOTONIC, &tp);
      offset = (unsigned int) tp.tv_nsec;

      CUDA_KERNEL_CALL(CacheSampleCSRSpMM_v3, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s, p, offset,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
  }
  
  cudaStreamSynchronize(thr_entry->stream);
}

template <typename IdType, typename DType>
void XSampleCsrmm(
  int m, int n, int s, int p,
  const IdType* A_indptr,
  const IdType* A_indices,
  const DType* B_data, DType* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  LOG(FATAL) << "Not supported yet";
}

template <>
void XSampleCsrmm<int32_t, float>(
  int m, int n, int s, int p,
  const int32_t* A_indptr,
  const int32_t* A_indices,
  const float* B_data, float* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#ifdef USE_FASTRAND
  CUDA_KERNEL_CALL(SampleSpMM_FastRand<int32_t>, 
    grid, block, shmem, 
    thr_entry->stream, 
    m, n, s, p,
    A_indptr,
    A_indices,
    B_data, C_data);
#endif
/*#ifdef USE_BUCKET
  CUDA_KERNEL_CALL(CacheSampleSpMM_Bucket<int32_t>, 
    grid, block, shmem, thr_entry->stream, 
    m, n, s,
    A_indptr,
    A_indices,
    B_data, C_data);
#endif*/
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
