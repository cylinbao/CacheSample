// Uncomment this line to choose between FastRand or Bucket 
#define USE_FASTRAND
// #define USE_BUCKET

namespace dgl {

using namespace cuda;

// Bucket sampling. Take the first S items per row.
__global__ void CacheSample_CSRSpMM_v0(
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

// Stride sampling. Multiply the index with a fixed prime number.
__global__ void CacheSample_CSRSpMM_v1(
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

// Stride sampling as V1, but the prime number is a parameter.
__global__ void CacheSample_CSRSpMM_v2(
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

// V3. Use a more advanced random function for sampling
__global__ void CacheSample_CSRSpMM_v3(
  const int norm_bias, const int s, const int p, const int o,
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

// V4. Similar with V3, but make S not related to the actual shared memory size.
__global__ void CacheSample_CSRSpMM_v4(
  const int norm_bias, const int s, const int p, const int o,
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int sm_offset = threadIdx.y*blockDim.x;

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

    int ss = threadIdx.x;
    int ptr = lb + threadIdx.x;
    int r, x;

    int nout = (k - cid + blockDim.x - 1) / blockDim.x;
    for (int jj=lb; jj < hb && jj < (lb + s); jj+=blockDim.x) {
      if (ss < hb) {
        if (nnz < s) {
          sh[sm_offset + ss] = A_indices[ptr]*k;
        }
        else {
          r = ((ptr*ptr) + o) % p;
          x = (ptr <= p / 2) ? r : p - r;
          offset = lb + (x % nnz);
          sh[sm_offset + ss] = A_indices[offset]*k;
        }
      }
      __syncthreads();
      ptr += blockDim.x;

      for (int kk=0; (kk < blockDim.x) && (jj+kk < hb); kk++) {
        offset = sh[(sm_offset + kk)] + cid;
        if (nout > 0) 
          acc += B[offset];
      }
      __syncthreads();
    }
    offset = rid*k + cid;
    if (nout > 0)
      C[offset] = acc/norm;
  }
}

// CacheSample2!! V0. Take a fixed faction of each row. Don't cache.
__global__ void CacheSample2_CSRSpMM_v0(
  const int p, const int o, const float sample_rate,
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int sm_offset = threadIdx.y*blockDim.x;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    float acc = 0.0;
    int nnz = hb - lb;
    int s_nnz = int(ceilf(nnz * sample_rate));

    int ptr, r, x;

    int nout = (k - cid + blockDim.x - 1) / blockDim.x;
    for (int kk = 0; kk < s_nnz; kk++) {
      r = ((kk*kk) + o) % p;
      x = (kk <= p / 2) ? r : p - r;
      ptr = lb + (x % nnz);
      offset = A_indices[ptr]*k + cid;
      acc += B[offset];
    }
    offset = rid*k + cid;
    if (nout > 0)
      C[offset] = acc/s_nnz;
  }
}

// CacheSample2!! v1, starts caching
__global__ void CacheSample2_CSRSpMM_v1(
  const int p, const int o, const float sample_rate,
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int sm_offset = threadIdx.y*blockDim.x;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  int offset, ptr, r, x;
  float acc = 0.0;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int nnz = hb - lb;
    int s_nnz = int(ceilf(nnz * sample_rate));
    int s_hb = lb + s_nnz;

    int nout = (k - cid + blockDim.x - 1) / blockDim.x;
    for (int jj = lb; jj < s_hb; jj += blockDim.x) {
      ptr = jj + threadIdx.x;
      r = ((ptr*ptr) + o) % p;
      x = (ptr <= p / 2) ? r : p - r;
      offset = lb + (x % nnz);
      sh[sm_offset + threadIdx.x] = A_indices[offset]*k;
      __syncthreads();

      for (int kk = 0; (kk < blockDim.x) && (jj+kk < s_hb); kk++) {
        offset = sh[(sm_offset + kk)] + cid;
        if (nout > 0) 
          acc += B[offset];
      }
      __syncthreads();
    }
    offset = rid*k + cid;
    if (nout > 0)
      C[offset] = acc/s_nnz;
  }
}

// CacheSample2!! v2, starts caching, force adding self-loop
__global__ void CacheSample2_CSRSpMM_v2(
  const int p, const int o, const float sample_rate,
  const int m, const int k, 
  const int32_t* A_indptr, const int32_t* A_indices,
  const float* B, float* C)
{
  extern __shared__ int sh[];
  int sm_offset = threadIdx.y*blockDim.x;

  int cid = blockIdx.y*blockDim.x + threadIdx.x;
  int rid = blockIdx.x*blockDim.y + threadIdx.y;

  int offset, ptr, r, x;
  float acc = 0.0;

  if (rid < m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int nnz = hb - lb;
    int s_nnz = int(ceilf(nnz * sample_rate));
    int s_hb = lb + s_nnz;

    int nout = (k - cid + blockDim.x - 1) / blockDim.x;
    for (int jj = lb; jj < s_hb; jj += blockDim.x) {
      ptr = jj + threadIdx.x;
      r = ((ptr*ptr) + o) % p;
      x = (ptr <= p / 2) ? r : p - r;
      offset = lb + (x % nnz);
      sh[sm_offset + threadIdx.x] = A_indices[offset]*k;
      __syncthreads();

      for (int kk = 0; (kk < blockDim.x) && (jj+kk < s_hb); kk++) {
        offset = sh[(sm_offset + kk)] + cid;
        if (nout > 0) 
          acc += B[offset];
      }
      __syncthreads();
    }
    offset = rid*k + cid;
    if (nout > 0) {
      // add self-loop
      acc += B[offset];
      C[offset] = acc/(s_nnz+1);
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
  const int s, const int seed, 
  const float sample_rate,
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
  const int s, const int seed, 
  const float sample_rate,
  int m, int n, 
  const int32_t* A_indptr,
  const int32_t* A_indices,
  const float* B_data, float* C_data,
  dim3 grid, dim3 block,
  int shmem) {
  // LOG(INFO) << "Using kernel:" << kernel << ", sample_rate = " << sample_rate;
  int primes[10] = {577, 769, 983, 1193, 1429,
                    1619, 1871, 2089, 2339, 2579};
  int p = 21767;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  // if (kernel.length() == 11) {
  if (kernel == "CacheSample") {
    CUDA_KERNEL_CALL(CacheSample_CSRSpMM_v3, 
      grid, block, shmem, 
      thr_entry->stream, 
      norm_bias, s, p, seed,
      m, n, 
      A_indptr, A_indices,
      B_data, C_data);
  }
  else {
    // if (kernel.substr(11, 2) == "V0") {
    if (kernel == "CacheSample_V0") {
      CUDA_KERNEL_CALL(CacheSample_CSRSpMM_v0, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel == "CacheSample_V1") {
      CUDA_KERNEL_CALL(CacheSample_CSRSpMM_v1, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel == "CacheSample_V2") {
      p = primes[seed % 10];

      CUDA_KERNEL_CALL(CacheSample_CSRSpMM_v2, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s, p,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel == "CacheSample_V3") {
      CUDA_KERNEL_CALL(CacheSample_CSRSpMM_v3, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s, p, seed,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel == "CacheSample_V4") {
      CUDA_KERNEL_CALL(CacheSample_CSRSpMM_v4, 
        grid, block, shmem, 
        thr_entry->stream, 
        norm_bias, s, p, seed,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel == "CacheSample2_V0") {
      CUDA_KERNEL_CALL(CacheSample2_CSRSpMM_v0, 
        grid, block, shmem, 
        thr_entry->stream, 
        p, seed, sample_rate,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel == "CacheSample2_V1") {
      CUDA_KERNEL_CALL(CacheSample2_CSRSpMM_v1, 
        grid, block, shmem, 
        thr_entry->stream, 
        p, seed, sample_rate,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else if (kernel == "CacheSample2_V2") {
      CUDA_KERNEL_CALL(CacheSample2_CSRSpMM_v2, 
        grid, block, shmem, 
        thr_entry->stream, 
        p, seed, sample_rate,
        m, n, 
        A_indptr, A_indices,
        B_data, C_data);
    }
    else {
      LOG(FATAL) << "Unsupported CacheSample Version.";
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
