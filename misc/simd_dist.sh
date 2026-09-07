#!/bin/bash

# Define the prefetch distances to test (in terms of floats: 8 floats = 32 bytes)
DISTANCES=(0 8 16 32 64 128 256)
MATRIX_SIZE=2048
OUTPUT_FILE="prefetch_cache_results.csv"

# Add L1_Loads and L1_HitRate(%) to the CSV header
echo "PrefetchDistance,Instructions,L1_Loads,L1_Misses,L1_HitRate_Pct,MPKI" > "$OUTPUT_FILE"

make clean 2>/dev/null

for D in "${DISTANCES[@]}"; do
    echo "Compiling and testing Prefetch Distance: $D floats"
	
    mkdir -p bin
    # Compile with the injected PREFETCH_DIST macro
    g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -DPREFETCH_DIST=$D \
        -Iinclude -Wall src/*.cpp -o bin/matmul_bench
    
    # Run perf, explicitly capturing both L1 loads and L1 misses
    perf stat -e instructions,L1-dcache-loads,L1-dcache-load-misses \
        -o perf_tmp.txt \
        ./bin/matmul_bench simd $MATRIX_SIZE $MATRIX_SIZE $MATRIX_SIZE
    
    # Parse standard perf output formatting
    INSTRUCTIONS=$(awk '/instructions/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
    L1_LOADS=$(awk '/L1-dcache-loads/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
    L1_MISSES=$(awk '/L1-dcache-load-misses/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
    
    # Calculate Cache Hit Rate: 100 * (1 - (Misses / Loads))
    if [ ! -z "$L1_LOADS" ] && [ "$L1_LOADS" -gt 0 ] && [ ! -z "$L1_MISSES" ]; then
        HIT_RATE=$(echo "scale=4; 100 * (1 - ($L1_MISSES / $L1_LOADS))" | bc)
    else
        HIT_RATE="N/A"
    fi

    # Calculate Misses Per Kilo-Instruction (MPKI)
    if [ ! -z "$INSTRUCTIONS" ] && [ "$INSTRUCTIONS" -gt 0 ] && [ ! -z "$L1_MISSES" ]; then
        MPKI=$(echo "scale=4; ($L1_MISSES / ($INSTRUCTIONS / 1000))" | bc)
    else
        MPKI="N/A"
    fi
    
    # Write to CSV
    echo "$D,$INSTRUCTIONS,$L1_LOADS,$L1_MISSES,$HIT_RATE,$MPKI" >> "$OUTPUT_FILE"
done

rm -f perf_tmp.txt
echo "Sweep complete. Results saved to $OUTPUT_FILE."