#!/bin/bash

# Define the matrix sizes and tile sizes to test
MATRIX_SIZES=(256 512 1024 2048)
TILE_SIZES=(8 16 32 64 128)
KERNEL_SIZE=3
SEED=42
OUTPUT_FILE="tiling_results.csv"

echo "MatrixSize,TileSize,Instructions,L1_Misses,MPKI" > $OUTPUT_FILE

# Ensure the executable is built
make

for M in "${MATRIX_SIZES[@]}"; do
    for T in "${TILE_SIZES[@]}"; do
        echo "Testing Matrix: ${M}x${M}, Tile: ${T}x${T}"
        
        # Execute bin/conv with the correct argument order:
        # ./bin/conv <stage> <H> <W> <K> <seed> <tile_h> <tile_w>
        perf stat -e instructions,L1-dcache-loads,L1-dcache-load-misses \
            -o perf_tmp.txt \
            ./bin/conv tile $M $M $KERNEL_SIZE $SEED $T $T
        
        # Parse perf output using awk
        INSTRUCTIONS=$(awk '/instructions/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
        L1_MISSES=$(awk '/L1-dcache-load-misses/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
        
        # Calculate MPKI
        if [ ! -z "$INSTRUCTIONS" ] && [ "$INSTRUCTIONS" -gt 0 ]; then
            MPKI=$(echo "scale=4; ($L1_MISSES / ($INSTRUCTIONS / 1000))" | bc)
        else
            MPKI="N/A"
        fi
        
        echo "$M,$T,$INSTRUCTIONS,$L1_MISSES,$MPKI" >> $OUTPUT_FILE
    done
done

rm perf_tmp.txt
echo "Benchmarking complete. Results saved to $OUTPUT_FILE."