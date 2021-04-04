// #include "../../runtime/cuda/cuda_common.h"

// using namespace cuda;

__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0;
}

__global__ void topoCacheCoarsenSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {

    int cid = (blockIdx.y<<6)+threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid+1];
    int ptr = lb+threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          acc1 = sum_reduce(acc1, B[offset]);
          acc2 = sum_reduce(acc2, B[(offset+32)]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
      C[offset+32] = acc2;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout>1) {
          acc2 = sum_reduce(acc2, B[(offset+32)]);}
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));}
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
      if (nout>1) {
      C[offset+32] = acc2;}
    }
  }
} 

__global__ void topoCacheSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;
  
  int cid = (blockIdx.y<<5)+threadIdx.x;
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    int ptr = lb+threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[sm_offset+kk]+cid;
          acc1 = sum_reduce(acc1, B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
    }
  }
}

__global__ void topoSimpleSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      // offset = __ldg(A_indices+ptr)*k+threadIdx.x;
      // acc1 = sum_reduce(acc1, __ldg(B+offset));
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
  }
}


template <typename DType>
int XTopoCsrmm(//const RuntimeConfig& rtcfg,
  const DLContext& ctx,
  int m, int n,
  const int* A_indptr,
  const int* A_indices,
  const DType* B, DType* C) {
LOG(INFO) << "Not supported by custom spmm";
return -1;
}

template <>
int XTopoCsrmm<float>(//const RuntimeConfig& rtcfg,
  const DLContext& ctx,
  int m, int n,
  const int* A_indptr,
  const int* A_indices,
  const float* B, float* C) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  // LOG(INFO) << "Using custom spmm";
  if (n<32) {
    const int row_per_block = 128/n;
    const int n_block = (m+row_per_block-1)/row_per_block;
    // topoSimpleSPMMKernel<<< dim3(n_block,1,1),dim3(n, row_per_block, 1), 0, thr_entry->stream>>>(m,n,A_indptr,A_indices,B,C);

    CUDA_KERNEL_CALL(topoSimpleSPMMKernel, 
      dim3(n_block,1,1), dim3(n, row_per_block, 1), 0, 
      thr_entry->stream, 
      m, n,
      A_indptr,
      A_indices,
      B, C);
    return 0;
  }
  if (n<64) {
    const int tile_k = (n+31)/32;
    const int n_block = (m+3)/4;
    // topoCacheSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,4,1), 128*sizeof(int), rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);

    CUDA_KERNEL_CALL(topoCacheSPMMKernel, 
      dim3(n_block, tile_k, 1), dim3(32, 4, 1), 128*sizeof(int), 
      thr_entry->stream, 
      m, n,
      A_indptr,
      A_indices,
      B, C);
    return 0;
  }
  else {
    const int tile_k = (n+63)/64;
    const int n_block = (m+8-1)/8;
    // topoCacheCoarsenSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,8,1), 8*32*sizeof(int), rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);

    CUDA_KERNEL_CALL(topoCacheCoarsenSPMMKernel, 
      dim3(n_block, tile_k, 1), dim3(32, 8, 1), 8*32*sizeof(int), 
      thr_entry->stream, 
      m, n,
      A_indptr,
      A_indices,
      B, C);
    return 0;
  }
}

template <typename DType>
void GeSpmmCsrmm(
  // const RuntimeConfig& rtcfg,
  const DLContext& ctx,
  const aten::CSRMatrix& csr,
  const DType* B_data, DType* C_data,
  int x_length) {

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int m = csr.num_rows;
  const int n = x_length;
  typedef int32_t Idx;

  // int ret = XTopoCsrmm<DType> ( rtcfg,
  XTopoCsrmm<DType> (ctx,
    m, n, 
    static_cast<Idx*>(csr.indptr->data),
    static_cast<Idx*>(csr.indices->data),
    B_data, C_data
  );
  // if (ret == -1) {
  //   CusparseCsrmm2(rtcfg, csr, B_data, C_data, x_length);
  // }

  cudaStreamSynchronize(thr_entry->stream);
  CUDA_CALL(cudaGetLastError());
}
