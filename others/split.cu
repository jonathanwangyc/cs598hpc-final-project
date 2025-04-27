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
// New fields: col_chunk_offset, col_chunk_size, and rank index r
struct Block {
    int type;             // DENSE (0) or LOW_RANK (1)
    int row_offset;       // Global row start
    int col_offset;       // Global col start
    int m, n;             // dims of U (m×k) or slice of V (n×k)
    float* denseData;     // if DENSE
    float* U;             // if LOW_RANK: full U pointer (m×k)
    float* V;             // if LOW_RANK: pointer into V slice (n×k)
    // --- new fields for per‑thread reduction ---
    int col_chunk_offset; // offset into V’s n columns for this chunk
    int col_chunk_size;   // how many columns in this chunk
    int r;                // which α[r] this chunk computes
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

int CHUNK_WIDTH = 256;

// ---------------------------------------------------------------------------
// Recursively compress a matrix A (dense representation stored in a vector<float> of size N x N)
// into a HODLR representation. row_offset and col_offset give the global position.
// s is the size of the current submatrix. max_level controls the recursion depth.
void compressHODLR(const std::vector<float>& A, int N,
    int row_off, int col_off, int s,
    int level, int max_level,
    std::vector<Block>& blocks)
{
    if(level == max_level || s <= globalKRank) {
    // (unchanged) store dense block
        Block D{DENSE, row_off, col_off, s, s, nullptr,nullptr,nullptr};
        D.denseData = new float[s*s];
        for(int i=0;i<s*s;i++){
            int r=i/s, c=i%s;
            D.denseData[i] = A[(row_off+r)*N + (col_off+c)];
        }
        blocks.push_back(D);
        return;
    }

    int mid = s/2;
    // --- Top-right low-rank block: size mid × (s-mid) ---
    {
        int rows = mid, cols = s-mid;
        auto sub = getSubMatrix(A, N, row_off, col_off+mid, rows, cols);
        std::vector<float> Ufull, Vfull;
        dummySVD(sub, rows, cols, globalKRank, Ufull, Vfull);

        // How many column-chunks to split into?
        int numColChunks = (cols + CHUNK_WIDTH-1) / CHUNK_WIDTH;
        for(int chunk=0; chunk<numColChunks; ++chunk){
            int c0 = chunk * CHUNK_WIDTH;
            int csize = std::min(CHUNK_WIDTH, cols - c0);
            // For each rank index r, emit one Block-task
            for(int r=0; r<globalKRank; ++r){
                Block B;
                B.type             = LOW_RANK;
                B.row_offset       = row_off;
                B.col_offset       = col_off + mid;
                B.m                = rows;     // full U has rows rows
                B.n                = csize;    // only csize columns of V
                B.U                = new float[rows * globalKRank];
                B.V                = new float[csize * globalKRank];
                // copy full U
                std::copy(Ufull.data(), Ufull.data() + rows*globalKRank, B.U);
                // copy only V rows [c0 .. c0+csize) for this chunk
                for(int i=0; i<csize; ++i)
                    for(int j=0; j<globalKRank; ++j)
                        B.V[i*globalKRank + j] = Vfull[(c0 + i)*globalKRank + j];
                // record new fields
                B.col_chunk_offset = c0;
                B.col_chunk_size   = csize;
                B.r                = r;
            blocks.push_back(B);
            }
        }
    }

    // --- Bottom-left low-rank block: size (s-mid) × mid ---
    {
        int rows = s-mid, cols = mid;
        auto sub = getSubMatrix(A, N, row_off+mid, col_off, rows, cols);
        std::vector<float> Ufull, Vfull;
        dummySVD(sub, rows, cols, globalKRank, Ufull, Vfull);

        int numColChunks = (cols + CHUNK_WIDTH-1) / CHUNK_WIDTH;
        for(int chunk=0; chunk<numColChunks; ++chunk){
            int c0 = chunk * CHUNK_WIDTH;
            int csize = std::min(CHUNK_WIDTH, cols - c0);
            for(int r=0; r<globalKRank; ++r){
                Block B;
                B.type             = LOW_RANK;
                B.row_offset       = row_off + mid;
                B.col_offset       = col_off;
                B.m                = rows;
                B.n                = csize;
                B.U                = new float[rows * globalKRank];
                B.V                = new float[csize * globalKRank];
                std::copy(Ufull.data(), Ufull.data() + rows*globalKRank, B.U);
                for(int i=0; i<csize; ++i)
                    for(int j=0; j<globalKRank; ++j)
                        B.V[i*globalKRank + j] = Vfull[(c0 + i)*globalKRank + j];
                B.col_chunk_offset = c0;
                B.col_chunk_size   = csize;
                B.r                = r;
                blocks.push_back(B);
            }
        }
    }

    // Recurse on diagonal quadrants
    compressHODLR(A, N, row_off,       col_off,    mid,   level+1, max_level, blocks);
    compressHODLR(A, N, row_off+mid,   col_off+mid, s-mid, level+1, max_level, blocks);
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
// New kernel: 1 block per row‑task, blockDim.x threads collaborate on columns/chunks.
__global__ void hodlr_mvm_kernel(const Block* blocks,
    const int* blockPrefix, int totalWork,
    const float* __restrict__ x,
    float* __restrict__ y, int k)
{
int taskId = blockIdx.x; 
if (taskId >= totalWork) return;

// 1) Identify which HODLR block and which local row
int low=0, high=totalWork; 
// binary search over blockPrefix to map taskId→(b,localRow)
while (low < high) {
int mid = (low + high) >> 1;
if (blockPrefix[mid+1] <= taskId) low = mid+1;
else                               high = mid;
}
int b = low;
int localRow = taskId - blockPrefix[b];
const Block &B = blocks[b];

extern __shared__ float sdata[];  // shared buffer for partial sums

if (B.type == DENSE) {
// 2) Partial dot for dense block: row×vector
float partial = 0.0f;
for (int j = threadIdx.x; j < B.n; j += blockDim.x) {
partial += B.denseData[localRow * B.n + j]
* x[B.col_offset + j];
}
// 3) Shared‑memory reduction (tree)  [oai_citation_attribution:2‡Home](https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf?utm_source=chatgpt.com)
sdata[threadIdx.x] = partial;
__syncthreads();
for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
if (threadIdx.x < s) {
sdata[threadIdx.x] += sdata[threadIdx.x + s];
}
__syncthreads();
}
// 4) Thread 0 writes result
if (threadIdx.x == 0) {
atomicAdd(&y[B.row_offset + localRow], sdata[0]);
}
}
else {
// LOW_RANK: each Block now encodes one (column‑chunk, rank=r)
int r = B.r;
// 2a) Partial dot for Vᵀ x over the chunk  [oai_citation_attribution:3‡NVIDIA Developer](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/?utm_source=chatgpt.com)
float pdot = 0.0f;
for (int j = threadIdx.x; j < B.col_chunk_size; j += blockDim.x) {
pdot += B.V[(j + B.col_chunk_offset) * k + r]
* x[B.col_offset + B.col_chunk_offset + j];
}
// 2b) Reduce across threads in the block
sdata[threadIdx.x] = pdot;
__syncthreads();
for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
if (threadIdx.x < s) {
sdata[threadIdx.x] += sdata[threadIdx.x + s];
}
__syncthreads();
}
// 3) Thread 0 multiplies by U and accumulates into y  [oai_citation_attribution:4‡Stack Overflow](https://stackoverflow.com/questions/49163482/cuda-reduction-warp-unrolling-school?utm_source=chatgpt.com)
if (threadIdx.x == 0) {
float dot = sdata[0];
float val = B.U[localRow * k + r] * dot;
atomicAdd(&y[B.row_offset + localRow], val);
}
}
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

    int threadsPerBlock = 256;  // or tune for occupancy
    int numBlocksGrid   = totalWork;      

    // Shared memory per block, one float per thread:
    size_t sharedBytes = threadsPerBlock * sizeof(float);
    
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
    
    // int threadsPerBlock = 256;
    // int numKernelBlocks = (totalWork + threadsPerBlock - 1) / threadsPerBlock;
    // std::cout << "Number of thread blocks launched: " << numKernelBlocks << "\n";
    
    // CUDA timing events.
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, 0));
    
    // Launch the kernel: pass globalKRank as the rank parameter.
    hodlr_mvm_kernel<<<numBlocksGrid, threadsPerBlock, sharedBytes>>>(
        d_blocks,          // device Block*
        d_blockPrefix,     // device prefix sums
        totalWork,         // total number of rows
        d_x,               // input vector
        d_y, globalKRank                // output vector (zeroed)
    );
    
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
    int warmup_iterations = 10;
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
    int test_iterations = 100;
    cudaEvent_t startCUBLAS, stopCUBLAS;
    CUDA_CHECK(cudaEventCreate(&startCUBLAS));
    CUDA_CHECK(cudaEventCreate(&stopCUBLAS));
    CUDA_CHECK(cudaEventRecord(startCUBLAS, 0));
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