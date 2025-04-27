#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error checks
#define CUDA_CHECK(call)                                                   \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)                \
                << " (" << __FILE__ << ":" << __LINE__ << ")\n";            \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while (0)

#define CUBLAS_CHECK(call)                                                 \
  do {                                                                      \
    cublasStatus_t st = call;                                               \
    if (st != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS Error: " << st                                   \
                << " (" << __FILE__ << ":" << __LINE__ << ")\n";            \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while (0)

// Fill a host array with random floats in [-1,1].
void fillRandom(std::vector<float>& v) {
    for (auto& x : v) x = (2.0f * rand() / RAND_MAX) - 1.0f;
}

__global__ void computeAlphaKernel(
    const float * __restrict__ V,
    const float * __restrict__ x,
    float * __restrict__ alpha,
    int n, int k)
{
    int r = blockIdx.x;            // one block per rank
    if (r >= k) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    // stride across the length-n row
    for (int idx = tid; idx < n; idx += blockDim.x) {
        sum += V[r * n + idx] * x[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // parallel reduction within block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        alpha[r] = sdata[0];
    }
}

// Kernel 2: y[i] = ∑_{r=0..k-1} U[i*k + r] * alpha[r]
__global__ void computeYKernel(
    const float * __restrict__ U,
    const float * __restrict__ alpha,
    float * __restrict__ y,
    int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    for (int r = 0; r < k; ++r) {
        sum += U[i * k + r] * alpha[r];
    }
    y[i] = sum;
}

int main(int argc, char* argv[]) {
    // Parse n and k (defaults if not provided)
    int n = 32768/(2^2);
    int k = 15;
    if (argc >= 2) n = std::atoi(argv[1]);
    if (argc >= 3) k = std::atoi(argv[2]);
    std::cout << "Running n=" << n << ", k=" << k << "\n";

    // Host allocations
    std::vector<float> hA(n*n), hU(n*k), hV(k*n), hx(n);
    fillRandom(hA);
    fillRandom(hU);
    fillRandom(hV);
    fillRandom(hx);


    std::vector<float> A_col(n*n);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            A_col[j*n + i] = hA[i*n + j];

    // Device allocations
    float *dA, *dU, *dV, *dx, *dy, *dalpha;
    CUDA_CHECK(cudaMalloc(&dA,  n*n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dU,  n*k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV,  k*n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dx,     n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy,     n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dalpha, k * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dA,  A_col.data(),  n*n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dU,  hU.data(),  n*k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,  hV.data(),  k*n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx,  hx.data(),     n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dy, 0,            n * sizeof(float)));  // zero out output


    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    float ms_dense = 0.0f, ms_lr = 0.0f, ms_cuda = 0.0f;

    // --- Dense MVM timing ---
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemv(handle,
                             CUBLAS_OP_N,
                             n,      // rows
                             n,      // cols
                             &alpha,
                             dA,
                             n,      // leading dim
                             dx,
                             1,
                             &beta_zero,
                             dy,
                             1));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_dense, start, stop));

    // Zero dy again before low-rank
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));

    // --- Low-rank MVM timing ---
    CUDA_CHECK(cudaEventRecord(start));
    // 1) alpha = Vᵀ · x   (V is k×n, stored row-major as V[k][n])
    CUBLAS_CHECK(cublasSgemv(handle,
                             CUBLAS_OP_T,
                             k,      // rows of V
                             n,      // cols of V
                             &alpha,
                             dV,
                             k,      // leading dim = n
                             dx,
                             1,
                             &beta_zero,
                             dalpha,
                             1));
    // 2) y += U · alpha   (U is n×k)
    CUBLAS_CHECK(cublasSgemv(handle,
                             CUBLAS_OP_N,
                             n,      // rows of U
                             k,      // cols of U
                             &alpha,
                             dU,
                             n,      // leading dim = k
                             dalpha,
                             1,
                             &beta_one,
                             dy,
                             1));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_lr, start, stop));

    CUDA_CHECK(cudaMemset(dy, 0, n*sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    // launch alpha kernel: k blocks, 256 threads, shared=256*sizeof(float)
    computeAlphaKernel<<<k, 256, 256 * sizeof(float)>>>(dV, dx, dalpha, n, k);
    // then y kernel: ceil(n/256) blocks, 256 threads
    int blocksY = (n + 255) / 256;
    computeYKernel<<<blocksY, 256>>>(dU, dalpha, dy, n, k);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_cuda, start, stop));

    // Print timings
    std::cout << "Dense MVM (cublasSgemv, nxn):       " << ms_dense << " ms\n";
    std::cout << "Low-rank MVM (2x cublasSgemv):      " << ms_lr    << " ms\n";
    std::cout << "Low-rank MVM (custom kernels):      " << ms_cuda    << " ms\n";

    // Cleanup
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dU); cudaFree(dV);
    cudaFree(dx); cudaFree(dy); cudaFree(dalpha);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}