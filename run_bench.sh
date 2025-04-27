#!/usr/bin/env bash
set -euo pipefail

# Build (adjust to your build system)
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j

BIN=./hodlr_mvm        # adjust to your binary name
OUT=./results.csv
RUNS=10                # number of repetitions per (n,k)

# CSV header
echo "n,k,cpu_ms,kernel_ms,total_ms,gflops,rel_error" > "$OUT"

SIZES=(1024 2048 4096 8192 16384)
KS=(5 10 15 20)

for n in "${SIZES[@]}"; do
  # compute a reasonable max_level for HODLR; e.g. log2(n)-1
  lvl=$(echo "l($n)/l(2)-1" | bc -l)
  max_level=${lvl%.*}

  for k in "${KS[@]}"; do
    echo "→ benchmarking n=$n, k=$k  (averaging over $RUNS runs)"

    # accumulators
    sum_cpu=0
    sum_kernel=0
    sum_total=0
    sum_gflops=0
    sum_err=0

    for run in $(seq 1 $RUNS); do
      # one single-record CSV line from your program:
      record=$("$BIN" "$n" "$max_level" "$k")
      # split into fields
      IFS=',' read -r rn rk cpu_ms kernel_ms total_ms gflops relerr <<< "$record"

      sum_cpu=$(echo "$sum_cpu + $cpu_ms" | bc)
      sum_kernel=$(echo "$sum_kernel + $kernel_ms" | bc)
      sum_total=$(echo "$sum_total + $total_ms" | bc)
      sum_gflops=$(echo "$sum_gflops + $gflops" | bc)
      sum_err=$(echo "$sum_err + $relerr" | bc)
    done

    # compute averages (with 3 decimal places)
    avg_cpu=$(echo "scale=3; $sum_cpu / $RUNS" | bc)
    avg_kernel=$(echo "scale=3; $sum_kernel / $RUNS" | bc)
    avg_total=$(echo "scale=3; $sum_total / $RUNS" | bc)
    avg_gflops=$(echo "scale=3; $sum_gflops / $RUNS" | bc)
    avg_err=$(echo "scale=3; $sum_err / $RUNS" | bc)

    # append to CSV
    echo "$n,$k,$avg_cpu,$avg_kernel,$avg_total,$avg_gflops,$avg_err" >> "$OUT"
  done
done

echo "→ done. aggregated results in $OUT"