#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro to check CUDA API calls.
#define CUDA_CHECK(call)                                        \
    {                                                           \
        cudaError_t err = call;                                 \
        if (err != cudaSuccess) {                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                          \
        }                                                       \
    }

// Global variable for the fixed rank approximation.
// Its value will be set from the configuration file.
int globalKRank = 0;

// Enumeration for block type.
enum BlockType { DENSE = 0, LOW_RANK = 1 };

// Host structure for a HODLR block.
struct Block {
    int type;         // DENSE (0) or LOW_RANK (1)
    int row_offset;   // Global row start index in the full matrix.
    int col_offset;   // Global col start index.
    int m;            // Number of rows in this block.
    int n;            // Number of columns in this block.
    // For dense blocks: pointer to host-allocated memory (row-major, size m*n)
    float* denseData;
    // For low-rank blocks: pointers to host-allocated arrays U (m x globalKRank) and V (n x globalKRank)
    float* U;
    float* V;
};

// ---------------------------------------------------------------------------
// Helper: generate a random dense matrix stored in row-major order.
std::vector<float> generateRandomMatrix(int n) {
    std::vector<float> A(n * n);
    for (int i = 0; i < n * n; i++) {
        // Random numbers in [-1,1]
        A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    return A;
}

// Helper: print a dense matrix given as a vector<float>.
void printDenseMatrix(const std::vector<float>& mat, int rows, int cols, const std::string &name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        std::cout << "[ ";
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << mat[i*cols+j] << " ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

// Helper: print a vector.
void printVector(const std::vector<float>& vec, const std::string &name) {
    std::cout << name << ": [ ";
    for (auto v : vec)
        std::cout << std::setw(8) << v << " ";
    std::cout << "]\n";
}

double compute2NormDifference(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vectors must be of the same size." << std::endl;
        return -1.0;
    }
    double sumSq = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        double diff = static_cast<double>(v1[i] - v2[i]);
        sumSq += diff * diff;
    }
    return std::sqrt(sumSq);
}

double computeRelative2NormError(const std::vector<float>& v_ref,
                                 const std::vector<float>& v_diff) {
    double diffNorm = compute2NormDifference(v_ref, v_diff);
    double refNorm = compute2NormDifference(v_ref, std::vector<float>(v_ref.size(), 0.0f));
    return diffNorm / refNorm;
}

// Helper: extract a submatrix from a dense matrix A (stored in row-major order).
// Returns a vector representing the submatrix of size m x n starting at (row_offset, col_offset)
std::vector<float> getSubMatrix(const std::vector<float>& A, int N, int row_offset, int col_offset, int m, int n) {
    std::vector<float> B(m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            B[i*n + j] = A[(row_offset + i)*N + (col_offset + j)];
    return B;
}

// Dummy SVD: Instead of computing a proper SVD, this routine simply takes the first k columns
// for U and the first k rows for V. (Not a true SVD! For demonstration only.)
void dummySVD(const std::vector<float>& block, int m, int n, int rank,
              std::vector<float>& U, std::vector<float>& V) {
    int k_use = std::min({rank, m, n});
    U.resize(m * rank, 0.0f);
    V.resize(n * rank, 0.0f);
    // For U, copy the first k_use columns from the block.
    for (int i = 0; i < m; i++) {
        for (int r = 0; r < k_use; r++) {
            U[i*rank + r] = block[i*n + r];
        }
    }
    // For V, copy the first k_use rows of the block.
    for (int j = 0; j < n; j++) {
        for (int r = 0; r < k_use; r++) {
            V[j*rank + r] = block[r*n + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Recursively compress a matrix A (dense representation stored in a vector<float> of size N x N)
// into a HODLR representation. row_offset and col_offset give the global position.
// s is the size of the current submatrix. max_level controls the recursion depth.
void compressHODLR(const std::vector<float>& A, int N,
                   int row_offset, int col_offset, int s,
                   int level, int max_level, std::vector<Block>& blocks) {
    // If at maximum level or if the block is very small, store dense.
    if(level == max_level || s <= globalKRank) {
        Block block;
        block.type = DENSE;
        block.row_offset = row_offset;
        block.col_offset = col_offset;
        block.m = s;
        block.n = s;
        block.denseData = new float[s * s];
        for (int i = 0; i < s*s; i++) {
            int r = i / s, c = i % s;
            block.denseData[i] = A[(row_offset + r) * N + (col_offset + c)];
        }
        block.U = nullptr; 
        block.V = nullptr;
        blocks.push_back(block);
        return;
    }
    int mid = s / 2;
    // Compress off-diagonal blocks.
    // Top-right block: dimensions mid x (s-mid)
    {
        std::vector<float> subBlock = getSubMatrix(A, N, row_offset, col_offset+mid, mid, s-mid);
        std::vector<float> U, V;
        dummySVD(subBlock, mid, s-mid, globalKRank, U, V);
        Block block;
        block.type = LOW_RANK;
        block.row_offset = row_offset;
        block.col_offset = col_offset+mid;
        block.m = mid;
        block.n = s - mid;
        block.U = new float[mid * globalKRank];
        block.V = new float[(s - mid) * globalKRank];
        std::copy(U.begin(), U.end(), block.U);
        std::copy(V.begin(), V.end(), block.V);
        block.denseData = nullptr;
        blocks.push_back(block);
    }
    // Bottom-left block: dimensions (s-mid) x mid
    {
        std::vector<float> subBlock = getSubMatrix(A, N, row_offset+mid, col_offset, s-mid, mid);
        std::vector<float> U, V;
        dummySVD(subBlock, s-mid, mid, globalKRank, U, V);
        Block block;
        block.type = LOW_RANK;
        block.row_offset = row_offset+mid;
        block.col_offset = col_offset;
        block.m = s - mid;
        block.n = mid;
        block.U = new float[(s - mid) * globalKRank];
        block.V = new float[mid * globalKRank];
        std::copy(U.begin(), U.end(), block.U);
        std::copy(V.begin(), V.end(), block.V);
        block.denseData = nullptr;
        blocks.push_back(block);
    }
    // Recursively process the diagonal blocks.
    compressHODLR(A, N, row_offset, col_offset, mid, level+1, max_level, blocks);
    compressHODLR(A, N, row_offset+mid, col_offset+mid, s-mid, level+1, max_level, blocks);
}

// ---------------------------------------------------------------------------
// Assemble the HODLR representation into a full dense matrix (approximated) stored in row-major order.
std::vector<float> assembleDenseMatrix(const std::vector<Block>& blocks, int N) {
    std::vector<float> A_dense(N * N, 0.0f);
    for (const auto& block : blocks) {
        if(block.type == DENSE) {
            for (int i = 0; i < block.m; i++) {
                for (int j = 0; j < block.n; j++) {
                    int globalRow = block.row_offset + i;
                    int globalCol = block.col_offset + j;
                    A_dense[globalRow * N + globalCol] = block.denseData[i * block.n + j];
                }
            }
        } else { // LOW_RANK: approximate using U * V^T.
            for (int i = 0; i < block.m; i++) {
                for (int j = 0; j < block.n; j++) {
                    float sum = 0.0f;
                    for (int r = 0; r < globalKRank; r++) {
                        sum += block.U[i * globalKRank + r] * block.V[j * globalKRank + r];
                    }
                    int globalRow = block.row_offset + i;
                    int globalCol = block.col_offset + j;
                    A_dense[globalRow * N + globalCol] = sum;
                }
            }
        }
    }
    return A_dense;
}

// ---------------------------------------------------------------------------
// CUDA kernel: each thread computes one row of one block in the flattened work array.
// The kernel now takes an additional parameter 'k' for the rank.
__global__ void hodlr_mvm_kernel(const Block* blocks, int numBlocks,
                                 const int* blockPrefix, int totalWork,
                                 const float* x, float* y, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= totalWork) return;
    // Binary search over blockPrefix to find the corresponding block.
    int low = 0, high = numBlocks, b = -1;
    while(low < high) {
        int mid = (low + high) >> 1;
        if(blockPrefix[mid+1] <= tid)
            low = mid + 1;
        else
            high = mid;
    }
    b = low;
    int localRow = tid - blockPrefix[b];
    
    Block block = blocks[b];
    float result = 0.0f;
    if(block.type == DENSE) {
        for (int j = 0; j < block.n; j++)
            result += block.denseData[localRow * block.n + j] * x[block.col_offset + j];
    } else { // LOW_RANK
        for (int r = 0; r < k; r++) {
            float dot = 0.0f;
            for (int j = 0; j < block.n; j++) {
                dot += block.V[j * k + r] * x[block.col_offset + j];
            }
            result += block.U[localRow * k + r] * dot;
        }
    }
    // Use atomicAdd to accumulate if multiple blocks contribute to the same row.
    atomicAdd(&y[block.row_offset + localRow], result);
}

// ---------------------------------------------------------------------------
// CPU dense matrix–vector multiplication.
// A is a dense matrix stored in row-major order with dimensions N x N.
std::vector<float> cpuMVM(const std::vector<float>& A, const std::vector<float>& x, int N) {
    std::vector<float> y(N, 0.0f);
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++)
            sum += A[i * N + j] * x[j];
        y[i] = sum;
    }
    return y;
}

// Helper: Convert a matrix from row-major to column-major order.
std::vector<float> convertRowMajorToColMajor(const std::vector<float>& A, int N) {
    std::vector<float> A_col(A.size(), 0.0f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_col[j * N + i] = A[i * N + j];
        }
    }
    return A_col;
}

// ---------------------------------------------------------------------------
// Main function.
int main() {
    // ---------------------------
    // Read configuration from file "config.txt".
    // The file should contain three numbers: n, max_levels, and k_rank (in that order).
    int n = 0, max_levels = 0, fileKRank = 0;
    std::ifstream configFile("config.txt");
    if (!configFile) {
        std::cerr << "Error: Could not open config.txt" << std::endl;
        return 1;
    }
    configFile >> n >> max_levels >> fileKRank;
    if(n <= 0 || max_levels <= 0 || fileKRank <= 0) {
        std::cerr << "Error: Invalid configuration values." << std::endl;
        return 1;
    }
    globalKRank = fileKRank;
    std::cout << "Configuration:\n";
    std::cout << "Matrix dimension n = " << n << "\n";
    std::cout << "Max hierarchical levels = " << max_levels << "\n";
    std::cout << "Low-rank approximation k_rank = " << globalKRank << "\n";
    
    // ---------------------------
    // Generate random matrix A.
    std::cout << "Generating random " << n << "x" << n << " dense matrix A...\n";
    std::vector<float> A = generateRandomMatrix(n);
    #ifdef TEST
        printDenseMatrix(A, n, n, "Original matrix A");
    #endif
    
    // Compress A into a HODLR representation.
    std::vector<Block> blocks;
    compressHODLR(A, n, 0, 0, n, 0, max_levels, blocks);
    std::cout << "HODLR compression complete; number of blocks: " << blocks.size() << "\n";
    
    // Assemble the approximated dense matrix from HODLR blocks.
    std::vector<float> A_dense = assembleDenseMatrix(blocks, n);
    #ifdef TEST
        printDenseMatrix(A_dense, n, n, "Assembled dense matrix from HODLR");
    #endif
    
    // Generate input vector x (here, for simplicity, all 1's).
    std::vector<float> x(n, 1.0f);
    #ifdef TEST
        printVector(x, "Input vector x");
    #endif
    
    // ---------------------------
    // CPU MVM using the assembled dense matrix.
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> y_cpu = cpuMVM(A_dense, x, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    
    // ---------------------------
    // GPU MVM using the HODLR blocks.
    int numBlocks = blocks.size();
    std::vector<int> h_blockPrefix(numBlocks + 1, 0);
    for (int b = 0; b < numBlocks; b++) {
        h_blockPrefix[b+1] = h_blockPrefix[b] + blocks[b].m;
    }
    int totalWork = h_blockPrefix[numBlocks];
    
    // Prepare device copies of blocks.
    std::vector<Block> h_devBlocks(numBlocks);
    for (int b = 0; b < numBlocks; b++) {
        h_devBlocks[b].type = blocks[b].type;
        h_devBlocks[b].row_offset = blocks[b].row_offset;
        h_devBlocks[b].col_offset = blocks[b].col_offset;
        h_devBlocks[b].m = blocks[b].m;
        h_devBlocks[b].n = blocks[b].n;
        if (blocks[b].type == DENSE) {
            int size = blocks[b].m * blocks[b].n;
            CUDA_CHECK(cudaMalloc((void**)&h_devBlocks[b].denseData, size * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(h_devBlocks[b].denseData, blocks[b].denseData, size * sizeof(float), cudaMemcpyHostToDevice));
            h_devBlocks[b].U = nullptr;
            h_devBlocks[b].V = nullptr;
        } else {
            int sizeU = blocks[b].m * globalKRank;
            int sizeV = blocks[b].n * globalKRank;
            CUDA_CHECK(cudaMalloc((void**)&h_devBlocks[b].U, sizeU * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&h_devBlocks[b].V, sizeV * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(h_devBlocks[b].U, blocks[b].U, sizeU * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_devBlocks[b].V, blocks[b].V, sizeV * sizeof(float), cudaMemcpyHostToDevice));
            h_devBlocks[b].denseData = nullptr;
        }
    }
    Block* d_blocks;
    CUDA_CHECK(cudaMalloc((void**)&d_blocks, numBlocks * sizeof(Block)));
    CUDA_CHECK(cudaMemcpy(d_blocks, h_devBlocks.data(), numBlocks * sizeof(Block), cudaMemcpyHostToDevice));
    
    int* d_blockPrefix;
    CUDA_CHECK(cudaMalloc((void**)&d_blockPrefix, (numBlocks+1)*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_blockPrefix, h_blockPrefix.data(), (numBlocks+1)*sizeof(int), cudaMemcpyHostToDevice));
    
    float* d_x;
    CUDA_CHECK(cudaMalloc((void**)&d_x, n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    
    float* d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_y, n*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_y, 0, n*sizeof(float)));
    
    int threadsPerBlock = 256;
    int numKernelBlocks = (totalWork + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Number of thread blocks launched: " << numKernelBlocks << "\n";
    
    // CUDA timing events.
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    
    // Launch the kernel: pass globalKRank as the rank parameter.
    hodlr_mvm_kernel<<<numKernelBlocks, threadsPerBlock>>>(d_blocks, numBlocks, d_blockPrefix, totalWork, d_x, d_y, globalKRank);
    
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, startEvent, stopEvent));
    
    std::vector<float> y_gpu(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, n*sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------------------
    // Now, incorporate cuBLAS SGEMV on A_dense.
    // Convert A_dense from row-major to column-major (cuBLAS expects column-major).
    auto convertRowMajorToColMajor = [n](const std::vector<float>& A_rm) -> std::vector<float> {
        std::vector<float> A_col(A_rm.size(), 0.0f);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A_col[j * n + i] = A_rm[i * n + j];
            }
        }
        return A_col;
    };
    std::vector<float> A_dense_col = convertRowMajorToColMajor(A_dense);

    cudaEvent_t startCUBLAS, stopCUBLAS;
    CUDA_CHECK(cudaEventCreate(&startCUBLAS));
    CUDA_CHECK(cudaEventCreate(&stopCUBLAS));
    CUDA_CHECK(cudaEventRecord(startCUBLAS, 0));

    // Allocate device memory for cuBLAS SGEMV.
    float *d_A_cublas, *d_x_cublas, *d_y_cublas;
    CUDA_CHECK(cudaMalloc((void**)&d_A_cublas, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x_cublas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y_cublas, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A_cublas, A_dense_col.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_cublas, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y_cublas, 0, n * sizeof(float)));

    // Create cuBLAS handle.
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }
    
    float alpha = 1.0f, beta = 0.0f;
    
    // --- Warm Up Phase ---
    int warmup_iterations = 0;
    for (int i = 0; i < warmup_iterations; i++) {
        stat = cublasSgemv(handle, CUBLAS_OP_N, n, n, &alpha, d_A_cublas, n,
                           d_x_cublas, 1, &beta, d_y_cublas, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS sgemv warmup failed" << std::endl;
            return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // Reset the output vector on device.
    CUDA_CHECK(cudaMemset(d_y_cublas, 0, n * sizeof(float)));
    
    // --- Timing Phase ---
    int test_iterations = 1;
    // cudaEvent_t startCUBLAS, stopCUBLAS;
    // CUDA_CHECK(cudaEventCreate(&startCUBLAS));
    // CUDA_CHECK(cudaEventCreate(&stopCUBLAS));
    // CUDA_CHECK(cudaEventRecord(startCUBLAS, 0));
    for (int i = 0; i < test_iterations; i++) {
        stat = cublasSgemv(handle, CUBLAS_OP_N, n, n, &alpha, d_A_cublas, n,
                           d_x_cublas, 1, &beta, d_y_cublas, 1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS sgemv test iteration failed" << std::endl;
            return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaEventRecord(stopCUBLAS, 0));
    CUDA_CHECK(cudaEventSynchronize(stopCUBLAS));
    float totalTime;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, startCUBLAS, stopCUBLAS));
    float cuBLAS_time = totalTime / test_iterations;
    
    std::vector<float> y_cublas(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(y_cublas.data(), d_y_cublas, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Clean up cuBLAS resources.
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A_cublas));
    CUDA_CHECK(cudaFree(d_x_cublas));
    CUDA_CHECK(cudaFree(d_y_cublas));
    CUDA_CHECK(cudaEventDestroy(startCUBLAS));
    CUDA_CHECK(cudaEventDestroy(stopCUBLAS));
    
    // ---------------------------
    // Print complete results and timings.    
    #ifdef TEST
        std::cout << "\nCPU MVM result:\n";
        printVector(y_cpu, "y_cpu");
    #endif
    std::cout << "CPU computation time: " << cpu_time.count() << " ms\n";

    #ifdef TEST
        std::cout << "\ncuBLAS result:\n";
        printVector(y_cublas, "y_cublas");
    #endif
    std::cout << "cuBLAS SGEMV average time over " << test_iterations << " iterations: "<< cuBLAS_time << " ms\n";
    
    #ifdef TEST
        std::cout << "GPU MVM result:\n";
        printVector(y_gpu, "y_gpu");
        #endif
    std::cout << "GPU kernel time: " << gpu_time << " ms\n";
    
    double relError_custom = computeRelative2NormError(y_cpu, y_gpu);
    double relError_cublas = computeRelative2NormError(y_cpu, y_cublas);
    std::cout << "Relative 2-norm error (Custom GPU kernel vs CPU): " << relError_custom << "\n";
    std::cout << "Relative 2-norm error (cuBLAS vs CPU): " << relError_cublas << "\n";

    long long total_ops = 2LL * n * n;
    double gpu_seconds = gpu_time * 1e-3;
    double gflops = (double)total_ops / 1e9 / gpu_seconds;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << std::setw(10) << "Stage"
            << std::setw(15) << "Ops (x1e9)"
            << std::setw(15) << "Time (ms)"
            << std::setw(15) << "GFLOPs/s"
            << "\n";
    
    std::cout << std::setw(10) << "Total"
    << std::setw(15) << (total_ops / 1e9)
    << std::setw(15) << gpu_time
    << std::setw(15) << gflops
    << "\n\n";
    
    // ---------------------------
    // Clean up device memory.
    for (int b = 0; b < numBlocks; b++) {
        if (h_devBlocks[b].denseData)
            CUDA_CHECK(cudaFree(h_devBlocks[b].denseData));
        if (h_devBlocks[b].U)
            CUDA_CHECK(cudaFree(h_devBlocks[b].U));
        if (h_devBlocks[b].V)
            CUDA_CHECK(cudaFree(h_devBlocks[b].V));
    }
    CUDA_CHECK(cudaFree(d_blocks));
    CUDA_CHECK(cudaFree(d_blockPrefix));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    // Clean up host-allocated block data.
    for (auto &block : blocks) {
        if (block.type == DENSE)
            delete[] block.denseData;
        else {
            delete[] block.U;
            delete[] block.V;
        }
    }
    
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    
    return 0;
}