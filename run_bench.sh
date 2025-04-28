#!/usr/bin/env bash
set -euo pipefail

# -------------------
# CONFIG
# -------------------
BIN=./benchmark_nowarmup      # path to your binary
OUT=./results_nowarmup.csv
RUNS=8                        # repetitions per (n,level,k)

# CSV header (14 fields)
echo "n,max_levels,k, \
cpu_ms, \
cublas_kernel_ms, \
cublas_total_ms, \
gpu_kernel_ms, \
gpu_total_ms, \
ops_cublas, \
ops_gpu, \
gflops_cublas, \
gflops_gpu, \
relerr_cublas, \
relerr_custom" > "$OUT"

# parameter lists
SIZES=(128 256 512 1024 2048 4096 8192 16384 32768)
LEVELS=(2 4 6 8 10 12)
KS=(6 8 10 12 14 16 18)

# -------------------
# BENCHMARK LOOP
# -------------------
for n in "${SIZES[@]}"; do
  for lvl in "${LEVELS[@]}"; do
    for k in "${KS[@]}"; do
      echo "Benchmarking n=$n, max_levels=$lvl, k=$k (averaging over $RUNS runs)"

      # zero accumulators
      sum_cpu=0
      sum_ck=0
      sum_ct=0
      sum_gk=0
      sum_gt=0
      sum_ops_c=0
      sum_ops_g=0
      sum_gf_c=0
      sum_gf_g=0
      sum_re_cb=0
      sum_re_cu=0

      for run in $(seq 1 $RUNS); do
        # single CSV line from your program:
        record=$("$BIN" "$n" "$lvl" "$k")
        IFS=',' read -r rn r_lv r_k \
          cpu_ms \
          cbl_k_ms cbl_t_ms \
          gpu_k_ms gpu_t_ms \
          ops_c ops_g \
          gf_c gf_g \
          re_cb re_cu <<< "$record"

        sum_cpu=$(awk -v a="$sum_cpu"  -v b="$cpu_ms"   'BEGIN{ printf "%.6f", a+b }')
        sum_ck=$(awk -v a="$sum_ck"   -v b="$cbl_k_ms" 'BEGIN{ printf "%.6f", a+b }')
        sum_ct=$(awk -v a="$sum_ct"   -v b="$cbl_t_ms" 'BEGIN{ printf "%.6f", a+b }')
        sum_gk=$(awk -v a="$sum_gk"   -v b="$gpu_k_ms"  'BEGIN{ printf "%.6f", a+b }')
        sum_gt=$(awk -v a="$sum_gt"   -v b="$gpu_t_ms"  'BEGIN{ printf "%.6f", a+b }')
        sum_ops_c=$(awk -v a="$sum_ops_c" -v b="$ops_c" 'BEGIN{ printf "%.0f", a+b }')
        sum_ops_g=$(awk -v a="$sum_ops_g" -v b="$ops_g" 'BEGIN{ printf "%.0f", a+b }')
        sum_gf_c=$(awk -v a="$sum_gf_c" -v b="$gf_c"   'BEGIN{ printf "%.6f", a+b }')
        sum_gf_g=$(awk -v a="$sum_gf_g" -v b="$gf_g"   'BEGIN{ printf "%.6f", a+b }')
        sum_re_cb=$(awk -v a="$sum_re_cb" -v b="$re_cb" 'BEGIN{ printf "%.9f", a+b }')
        sum_re_cu=$(awk -v a="$sum_re_cu" -v b="$re_cu" 'BEGIN{ printf "%.9f", a+b }')
      done

      # compute averages (6 decimal places for times, GFLOPs; 9 for errors)
      avg_cpu=$(awk -v s="$sum_cpu" -v r="$RUNS" 'BEGIN{ printf "%.6f", s/r }')
      avg_ck=$(awk -v s="$sum_ck"  -v r="$RUNS" 'BEGIN{ printf "%.6f", s/r }')
      avg_ct=$(awk -v s="$sum_ct"  -v r="$RUNS" 'BEGIN{ printf "%.6f", s/r }')
      avg_gk=$(awk -v s="$sum_gk"  -v r="$RUNS" 'BEGIN{ printf "%.6f", s/r }')
      avg_gt=$(awk -v s="$sum_gt"  -v r="$RUNS" 'BEGIN{ printf "%.6f", s/r }')
      avg_ops_c=$(awk -v s="$sum_ops_c" -v r="$RUNS" 'BEGIN{ printf "%.0f", s/r }')
      avg_ops_g=$(awk -v s="$sum_ops_g" -v r="$RUNS" 'BEGIN{ printf "%.0f", s/r }')
      avg_gf_c=$(awk -v s="$sum_gf_c" -v r="$RUNS" 'BEGIN{ printf "%.6f", s/r }')
      avg_gf_g=$(awk -v s="$sum_gf_g" -v r="$RUNS" 'BEGIN{ printf "%.6f", s/r }')
      avg_re_cb=$(awk -v s="$sum_re_cb" -v r="$RUNS" 'BEGIN{ printf "%.9f", s/r }')
      avg_re_cu=$(awk -v s="$sum_re_cu" -v r="$RUNS" 'BEGIN{ printf "%.9f", s/r }')

      # append aggregated line
      echo "$n,$lvl,$k,$avg_cpu,$avg_ck,$avg_ct,$avg_gk,$avg_gt,$avg_ops_c,$avg_ops_g,$avg_gf_c,$avg_gf_g,$avg_re_cb,$avg_re_cu" \
        >> "$OUT"
    done
  done
done

echo "Done. aggregated results in $OUT"