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
#include <nvToolsExt.h>

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

// Fill a host array with random floats in [-1,1].
void generateRandomVector(std::vector<float>& v) {
    for (auto& x : v) x = (2.0f * rand() / RAND_MAX) - 1.0f;
}

// Print only the DENSE blocks, each as its own matrix:
void printDenseBlocks(const std::vector<Block>& blocks) {
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        const Block& b = blocks[idx];
        if (b.type != 0) continue;  // skip LOW_RANK

        std::cout << "Block #" << idx
                  << " at global (" << b.row_offset
                  << "," << b.col_offset
                  << "), size " << b.m << "×" << b.n
                  << ":\n";

        for (int i = 0; i < b.m; ++i) {
            std::cout << " [ ";
            for (int j = 0; j < b.n; ++j) {
                // row‑major indexing
                float val = b.denseData[i * b.n + j];
                std::cout << std::setw(8) << val << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n";
    }
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

// ---------------------------------------------------------------------------
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

// Recursively compress a matrix A (dense representation stored in a vector<float> of size N x N)
// into a HODLR representation. row_offset and col_offset give the global position.
// s is the size of the current submatrix. max_level controls the recursion depth.
void compressHODLR(const std::vector<float>& A, int N,
                   int row_offset, int col_offset, int s,
                   int level, int max_level, std::vector<Block>& dense_blocks, 
                   std::vector<Block>& low_rank_blocks, std::vector<Block>& all_blocks) {
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
        dense_blocks.push_back(block);
        all_blocks.push_back(block);
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
        low_rank_blocks.push_back(block);
        all_blocks.push_back(block);
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
        low_rank_blocks.push_back(block);
        all_blocks.push_back(block);
    }
    // Recursively process the diagonal blocks.
    compressHODLR(A, N, row_offset, col_offset, mid, level+1, max_level, dense_blocks, low_rank_blocks, all_blocks);
    compressHODLR(A, N, row_offset+mid, col_offset+mid, s-mid, level+1, max_level, dense_blocks, low_rank_blocks, all_blocks);
}

// Convert an r×c matrix 'row' (row‑major) → 'col' (column‑major).
void rowToColMajor(const float* row, float* col, int r, int c) {
    // r = #rows, c = #cols in the row‑major buffer
    for(int i = 0; i < r; ++i) {
        for(int j = 0; j < c; ++j) {
            // element at row[i][j] moves to col[j][i]
            col[j*r + i] = row[i*c + j];
        }
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Read configuration from file "config.txt".
    // The file should contain three numbers: n, max_levels, and k_rank (in that order).
    int n = 0, max_levels = 0, fileKRank = 0;
    std::ifstream configFile("config.txt");
    if (!configFile) {
        std::cerr << "Error: Could not open config.txt" << std::endl;
        return 1;
    }
    configFile >> n >> max_levels >> fileKRank;
    if (n <= 0 || max_levels <= 0 || fileKRank <= 0) {
        std::cerr << "Error: Invalid configuration values." << std::endl;
        return 1;
    }
    globalKRank = fileKRank;
    std::cout << "Configuration:\n";
    std::cout << "Matrix dimension n = " << n << "\n";
    std::cout << "Max hierarchical levels = " << max_levels << "\n";
    std::cout << "Low-rank approximation k_rank = " << globalKRank << "\n";
    
    // Generate random matrix A.
    std::cout << "Generating random " << n << "x" << n << " dense matrix A...\n";
    std::vector<float> A = generateRandomMatrix(n);

    // Generate input vector x
    std::vector<float> h_x(n, 1.0f);
    // generateRandomVector(x);
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float))); 
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compress A into a HODLR representation.
    std::vector<Block> all_blocks;
    std::vector<Block> dense_blocks;
    std::vector<Block> low_rank_blocks;
    compressHODLR(A, n, 0, 0, n, 0, max_levels, dense_blocks, low_rank_blocks, all_blocks);
    std::cout << "HODLR compression complete; number of dense blocks: " << dense_blocks.size() << "\n";
    std::cout << "HODLR compression complete; number of low-rank blocks: " << low_rank_blocks.size() << "\n";

    std::sort(
        low_rank_blocks.begin(),
        low_rank_blocks.end(),
        [](const Block &a, const Block &b) {
            return a.m > b.m;  // larger m first
        }
    );

    // printDenseBlocks(dense_blocks);
    // printVector(h_x, "Input vector h_x");
    
    // ---------------------------
    // Assemble the approximated dense matrix from HODLR blocks.
    std::vector<float> A_dense = assembleDenseMatrix(all_blocks, n);
    // printDenseMatrix(A_dense, n, n, "Assembled dense matrix from HODLR");
    
    // CPU MVM using the assembled dense matrix.
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> y_cpu = cpuMVM(A_dense, h_x, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    
    // ---------------------------
    const int warmup_iters = 0;
    for (int w = 0; w < warmup_iters; ++w) {
        std::vector<float> h_y(n, 0.0f);
        float* d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        int denseBatch = dense_blocks.size();
        std::vector<const float*> hA_dense(denseBatch), hx_dense(denseBatch);
        std::vector<float*> hy_dense(denseBatch);

        // --- 2) Upload each block’s matrix to device and collect device pointers ---
        //    We’ll store each A_i in its own device buffer devA[i]
        std::vector<float*> devA(denseBatch);
        for (int i = 0; i < denseBatch; i++) {
            auto &blk = dense_blocks[i];
            size_t bytes = blk.m * blk.n * sizeof(float);
            CUDA_CHECK(cudaMalloc(&devA[i], bytes));
            // copy from host denseData (row-major) → device
            CUDA_CHECK(cudaMemcpy(devA[i], blk.denseData, bytes,
                                cudaMemcpyHostToDevice));
        }

        for(int i = 0; i < denseBatch; ++i) {
            auto &blk = dense_blocks[i];
            hA_dense[i] = devA[i];                // device ptr to m×n
            hx_dense[i] = d_x + blk.col_offset;   // offset into x
            hy_dense[i] = d_y + blk.row_offset;   // offset into y
        }

        const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
        const float **dA_dense, **dx_dense;
        float **dy_dense;

        CUDA_CHECK(cudaMalloc(&dA_dense, denseBatch * sizeof(const float*)));  // device array for A_i pointers 
        CUDA_CHECK(cudaMalloc(&dx_dense, denseBatch * sizeof(const float*)));  // device array for x_i pointers
        CUDA_CHECK(cudaMalloc(&dy_dense, denseBatch * sizeof(float*)));        // device array for y_i pointers

        // 4) Copy pointer arrays to device:
        CUDA_CHECK(cudaMemcpy(dA_dense, hA_dense.data(), denseBatch * sizeof(const float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dx_dense, hx_dense.data(), denseBatch * sizeof(const float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy_dense, hy_dense.data(), denseBatch * sizeof(float*),       cudaMemcpyHostToDevice));

        // cuBLAS handle
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        CUBLAS_CHECK(cublasSgemvBatched(handle,
            CUBLAS_OP_T,
            /*m=*/dense_blocks[0].m,
            /*n=*/dense_blocks[0].n,
            &alpha,
            dA_dense,
            /*lda=*/dense_blocks[0].m,
            dx_dense,
            /*incx=*/1,
            &beta_one,
            dy_dense,
            /*incy=*/1,
            denseBatch));

        CUDA_CHECK(cudaDeviceSynchronize());

        // Process blocks in groups of identical (m,n) ---
        int B = low_rank_blocks.size();
        int i = 0;
        while (i < B) {
            int m = low_rank_blocks[i].m;
            int n = low_rank_blocks[i].n;
            int j = i+1;
            while (j < B && low_rank_blocks[j].m==m && low_rank_blocks[j].n==n) ++j;
            int batch = j - i;

            // Allocate device buffers for U and V and copy ---
            std::vector<float*> dU(batch), dV(batch);
            for (int t = 0; t < batch; ++t) {
                auto &blk = low_rank_blocks[i+t];
                std::vector<float> U_col(m * globalKRank), V_col(n * globalKRank);
                rowToColMajor(blk.U, U_col.data(), blk.m, globalKRank); // m×k → col‑major
                rowToColMajor(blk.V, V_col.data(), blk.n, globalKRank); // n×k → col‑major

                CUDA_CHECK(cudaMalloc(&dV[t], blk.n * globalKRank * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&dU[t], blk.m * globalKRank * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(dV[t], V_col.data(),
                                    blk.n*globalKRank*sizeof(float),
                                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(dU[t], U_col.data(),
                                    blk.m*globalKRank*sizeof(float),
                                    cudaMemcpyHostToDevice));
            }

            // Build pointer arrays for batched GEMV
            std::vector<const float*> hVptr(batch), hUptr(batch), hxptr(batch);
            std::vector<float*>       halpha(batch), hyptr(batch);
            // allocate one contiguous alpha buffer for this batch:
            float *d_alpha = nullptr;
            CUDA_CHECK(cudaMalloc(&d_alpha, batch * globalKRank * sizeof(float)));

            for (int t = 0; t < batch; ++t) {
                auto &blk = low_rank_blocks[i+t];
                hVptr[t]  = dV[t];
                hUptr[t]  = dU[t];
                hxptr[t]  = d_x + blk.col_offset;
                halpha[t] = d_alpha + t*globalKRank;
                hyptr[t]  = d_y + blk.row_offset;
            }

            // upload pointer arrays
            const float **dVptr=nullptr, **dUptr=nullptr, **dxptr=nullptr;
            float       **dAlphaPtr=nullptr, **dYptr=nullptr;
            CUDA_CHECK(cudaMalloc(&dVptr,    batch*sizeof(const float*)));
            CUDA_CHECK(cudaMalloc(&dUptr,    batch*sizeof(const float*)));
            CUDA_CHECK(cudaMalloc(&dxptr,    batch*sizeof(const float*)));
            CUDA_CHECK(cudaMalloc(&dAlphaPtr,batch*sizeof(float*)));
            CUDA_CHECK(cudaMalloc(&dYptr,    batch*sizeof(float*)));

            CUDA_CHECK(cudaMemcpy(dVptr,    hVptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dUptr,    hUptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dxptr,    hxptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dAlphaPtr,halpha.data(),   batch*sizeof(float*), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dYptr,    hyptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));

            // Stage 1: alpha = V^T * x[col_offset] ---
            CUBLAS_CHECK(cublasSgemvBatched(
                handle,
                CUBLAS_OP_T,
                n, globalKRank,
                &alpha,
                dVptr, n,
                dxptr, 1,
                &beta_zero,
                dAlphaPtr, 1,
                batch
            ));

            // tage 2: y[row_offset] += U * alpha ---
            CUBLAS_CHECK(cublasSgemvBatched(
                handle,
                CUBLAS_OP_N,
                m, globalKRank,
                &alpha,
                dUptr, m,
                dAlphaPtr, 1,
                &beta_one,
                dYptr, 1,
                batch
            ));

            CUDA_CHECK(cudaDeviceSynchronize());

            // --- 5) Cleanup this group ---
            for (int t = 0; t < batch; ++t) {
                cudaFree(dV[t]);
                cudaFree(dU[t]);
            }
            cudaFree(dVptr);
            cudaFree(dUptr);
            cudaFree(dxptr);
            cudaFree(dAlphaPtr);
            cudaFree(dYptr);
            cudaFree(d_alpha);

            i = j;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        // CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, n*sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaFree(d_y);
        cudaFree(dA_dense);
        cudaFree(dx_dense);
        cudaFree(dy_dense);
        for (int i = 0; i < denseBatch; i++) {
            cudaFree(devA[i]);
        }
        cublasDestroy(handle);
    }

    // ---------------------------
    cudaEvent_t start_total, stop_total;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventRecord(start_total, /*stream=*/0));

    std::vector<float> h_y(n, 0.0f);
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int denseBatch = dense_blocks.size();
    std::vector<const float*> hA_dense(denseBatch), hx_dense(denseBatch);
    std::vector<float*> hy_dense(denseBatch);

    // --- 2) Upload each block’s matrix to device and collect device pointers ---
    //    We’ll store each A_i in its own device buffer devA[i]
    std::vector<float*> devA(denseBatch);
    for (int i = 0; i < denseBatch; i++) {
        auto &blk = dense_blocks[i];
        size_t bytes = blk.m * blk.n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&devA[i], bytes));
        // copy from host denseData (row-major) → device
        CUDA_CHECK(cudaMemcpy(devA[i], blk.denseData, bytes,
                              cudaMemcpyHostToDevice));
    }

    long long total_ops = 0;
    for(int i = 0; i < denseBatch; ++i) {
        auto &blk = dense_blocks[i];
        hA_dense[i] = devA[i];                // device ptr to m×n
        hx_dense[i] = d_x + blk.col_offset;   // offset into x
        hy_dense[i] = d_y + blk.row_offset;   // offset into y
        total_ops += 2LL * blk.m * blk.n; 
    }

    const float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    const float **dA_dense, **dx_dense;
    float **dy_dense;

    CUDA_CHECK(cudaMalloc(&dA_dense, denseBatch * sizeof(const float*)));  // device array for A_i pointers 
    CUDA_CHECK(cudaMalloc(&dx_dense, denseBatch * sizeof(const float*)));  // device array for x_i pointers
    CUDA_CHECK(cudaMalloc(&dy_dense, denseBatch * sizeof(float*)));        // device array for y_i pointers

    // 4) Copy pointer arrays to device:
    CUDA_CHECK(cudaMemcpy(dA_dense, hA_dense.data(), denseBatch * sizeof(const float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx_dense, hx_dense.data(), denseBatch * sizeof(const float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy_dense, hy_dense.data(), denseBatch * sizeof(float*),       cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // const int warmup_iters = 5;       // how many warm-up loops to run
    // // 1) WARM-UP: do dense + low-rank loops but DON'T time them.
    // for (int w = 0; w < warmup_iters; ++w) {
    //     // Dense batch
    //     CUBLAS_CHECK(cublasSgemvBatched(handle,
    //         CUBLAS_OP_T, /*m=*/dense_blocks[0].m, /*n=*/dense_blocks[0].n,
    //         &alpha, dA_dense, dense_blocks[0].m,
    //         dx_dense, 1, &beta_one,
    //         dy_dense, 1, denseBatch));
    // }
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float)));

    float gpu_exec_time = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, /*stream=*/0));

    // nvtxRangePushA("Dense MVM");
    CUBLAS_CHECK(cublasSgemvBatched(handle,
        CUBLAS_OP_T,
        /*m=*/dense_blocks[0].m,
        /*n=*/dense_blocks[0].n,
        &alpha,
        dA_dense,
        /*lda=*/dense_blocks[0].m,
        dx_dense,
        /*incx=*/1,
        &beta_one,
        dy_dense,
        /*incy=*/1,
        denseBatch));
    // nvtxRangePop();

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_dense = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dense, start, stop));
    gpu_exec_time += ms_dense;

    std::cout << "kernel time for denseMMM: " << ms_dense << " ms\n";

    CUDA_CHECK(cudaDeviceSynchronize());

    // Process blocks in groups of identical (m,n) ---
    int B = low_rank_blocks.size();
    int i = 0;
    while (i < B) {
        int m = low_rank_blocks[i].m;
        int n = low_rank_blocks[i].n;
        int j = i+1;
        while (j < B && low_rank_blocks[j].m==m && low_rank_blocks[j].n==n) ++j;
        int batch = j - i;

        // Allocate device buffers for U and V and copy ---
        std::vector<float*> dU(batch), dV(batch);
        for (int t = 0; t < batch; ++t) {
            auto &blk = low_rank_blocks[i+t];
            std::vector<float> U_col(m * globalKRank), V_col(n * globalKRank);
            rowToColMajor(blk.U, U_col.data(), blk.m, globalKRank); // m×k → col‑major
            rowToColMajor(blk.V, V_col.data(), blk.n, globalKRank); // n×k → col‑major

            CUDA_CHECK(cudaMalloc(&dV[t], blk.n * globalKRank * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dU[t], blk.m * globalKRank * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dV[t], V_col.data(),
                                  blk.n*globalKRank*sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dU[t], U_col.data(),
                                  blk.m*globalKRank*sizeof(float),
                                  cudaMemcpyHostToDevice));

            total_ops += 2LL * blk.n * globalKRank;  // Vᵀ·x
            total_ops += 2LL * blk.m * globalKRank;
        }

        // Build pointer arrays for batched GEMV
        std::vector<const float*> hVptr(batch), hUptr(batch), hxptr(batch);
        std::vector<float*>       halpha(batch), hyptr(batch);
        // allocate one contiguous alpha buffer for this batch:
        float *d_alpha = nullptr;
        CUDA_CHECK(cudaMalloc(&d_alpha, batch * globalKRank * sizeof(float)));

        for (int t = 0; t < batch; ++t) {
            auto &blk = low_rank_blocks[i+t];
            hVptr[t]  = dV[t];
            hUptr[t]  = dU[t];
            hxptr[t]  = d_x + blk.col_offset;
            halpha[t] = d_alpha + t*globalKRank;
            hyptr[t]  = d_y + blk.row_offset;
        }

        // upload pointer arrays
        const float **dVptr=nullptr, **dUptr=nullptr, **dxptr=nullptr;
        float       **dAlphaPtr=nullptr, **dYptr=nullptr;
        CUDA_CHECK(cudaMalloc(&dVptr,    batch*sizeof(const float*)));
        CUDA_CHECK(cudaMalloc(&dUptr,    batch*sizeof(const float*)));
        CUDA_CHECK(cudaMalloc(&dxptr,    batch*sizeof(const float*)));
        CUDA_CHECK(cudaMalloc(&dAlphaPtr,batch*sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&dYptr,    batch*sizeof(float*)));

        CUDA_CHECK(cudaMemcpy(dVptr,    hVptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dUptr,    hUptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dxptr,    hxptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dAlphaPtr,halpha.data(),   batch*sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dYptr,    hyptr.data(),    batch*sizeof(float*), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start, /*stream=*/0));

        // nvtxRangePushA("Low-Rank MVM");
        // Stage 1: alpha = V^T * x[col_offset] ---
        CUBLAS_CHECK(cublasSgemvBatched(
            handle,
            CUBLAS_OP_T,
            n, globalKRank,
            &alpha,
            dVptr, n,
            dxptr, 1,
            &beta_zero,
            dAlphaPtr, 1,
            batch
        ));

        // tage 2: y[row_offset] += U * alpha ---
        CUBLAS_CHECK(cublasSgemvBatched(
            handle,
            CUBLAS_OP_N,
            m, globalKRank,
            &alpha,
            dUptr, m,
            dAlphaPtr, 1,
            &beta_one,
            dYptr, 1,
            batch
        ));
        // nvtxRangePop();

        CUDA_CHECK(cudaEventRecord(stop, /*stream=*/0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_lr = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_lr, start, stop));
        gpu_exec_time += ms_lr;

        std::cout << "kernel time for size " << m << ": " << ms_lr << " ms\n";

        CUDA_CHECK(cudaDeviceSynchronize());

        // --- 5) Cleanup this group ---
        for (int t = 0; t < batch; ++t) {
            cudaFree(dV[t]);
            cudaFree(dU[t]);
        }
        cudaFree(dVptr);
        cudaFree(dUptr);
        cudaFree(dxptr);
        cudaFree(dAlphaPtr);
        cudaFree(dYptr);
        cudaFree(d_alpha);

        i = j;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_total, /*stream=*/0));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    // // Compute elapsed time in ms
    float gpu_total_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_total_time, start_total, stop_total));

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, n*sizeof(float), cudaMemcpyDeviceToHost));

    // printVector(y_cpu, "Output vector y_cpu");
    // printVector(h_y, "Output vector h_y");
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(dA_dense);
    cudaFree(dx_dense);
    cudaFree(dy_dense);
    cublasDestroy(handle);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------
    // Print complete results and timings.    
    std::cout << "CPU computation time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU kernel execution time: " << gpu_exec_time << " ms\n";
    std::cout << "GPU total execution time: " << gpu_total_time << " ms\n";

    double relError_custom = computeRelative2NormError(y_cpu, h_y);
    std::cout << "Relative 2-norm error (Custom GPU kernel vs CPU): " << relError_custom << "\n";

    double gpu_seconds = gpu_exec_time * 1e-3;
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
    << std::setw(15) << gpu_exec_time
    << std::setw(15) << gflops
    << "\n\n";
    
    return 0;
}