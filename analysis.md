### Analysis



n,max_levels,k,

cublas_kernel_ms,
cublas_total_ms,

gpu_kernel_ms,
gpu_total_ms,

gflops_cublas,
gflops_gpu,

1. four variable plots
- show the cublas_kernel_ms, gpu_kernel_ms (with warmup)

2. three variable plots
- Fix n, find the relationship of max_level, k to the cublas_kernel_ms, gpu_kernel_ms (with warmup)
- Fix k, see how max_level relates to n
- Fix max_level, see how k relates to n

3. two variable plots
- Fix max_level & k, for each n, show the cublas_kernel_ms, gpu_kernel_ms (with/without warmup)
- Fix max_level & k, for each n, show the gflops_cublas, gflops_gpu (with/without warmup)
- Fix max_level & k, for each n, show the cublas_total_ms, gpu_total_ms (with/without warmup)

4. other two variable plots
- For each n (group by n), show the highest recorded cpu_ms, cublas_total_ms, gpu_total_ms (with/without warmup)
- For each n (group by n), show the highest recorded relerr_cublas, relerr_custom