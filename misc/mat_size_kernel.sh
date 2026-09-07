#!/bin/bash

# Define matrix sizes to test for All
MATRIX_SIZE=2048
KERNEL_SIZES=(3 5 7 9)
SEED=42

# Optional command-line argument for output filename (e.g., ./simd_sweep.sh simd_128.csv)
OUTPUT_FILE="${1:-all_results.csv}"

echo "MatrixSize,Instructions,L1_Misses,MPKI" > "$OUTPUT_FILE"

# Ensure the executable is built
make

for K in "${KERNEL_SIZES[@]}"; do
    echo "Testing Kernel: ${K}x${K} (ALL Stage)"
    
    # Execute bin/conv with SIMD stage:
    # ./bin/conv simd <H> <W> <K> <seed>
    perf stat -e instructions,L1-dcache-loads,L1-dcache-load-misses \
        -o perf_tmp.txt \
        ./bin/conv all $MATRIX_SIZE $MATRIX_SIZE $K $SEED
    
    # Parse perf output
    INSTRUCTIONS=$(awk '/instructions/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
    L1_MISSES=$(awk '/L1-dcache-load-misses/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
    
    # Calculate MPKI
    if [ ! -z "$INSTRUCTIONS" ] && [ "$INSTRUCTIONS" -gt 0 ]; then
        MPKI=$(echo "scale=4; ($L1_MISSES / ($INSTRUCTIONS / 1000))" | bc)
    else
        MPKI="N/A"
    fi
    
    echo "$M,$INSTRUCTIONS,$L1_MISSES,$MPKI" >> "$OUTPUT_FILE"
done

rm -f perf_tmp.txt
echo "Benchmarking complete. Results saved to $OUTPUT_FILE."