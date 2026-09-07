#!/bin/bash

# Matrix sizes to test
SIZES=(128 256 512 1024 2048)

# Implementations to test
STAGES=("simd") 

OUTPUT_FILE="cache_misses_results.txt"

echo "Compiling the code via Makefile..."
make clean
make

# Exit immediately if compilation fails
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Clear the previous results file
> "$OUTPUT_FILE"

echo "Running benchmark to measure L1D, L2, and LLC misses..."
echo "========================================================" | tee -a "$OUTPUT_FILE"

for S in "${SIZES[@]}"; do
    echo "=== Matrix Size: ${S}x${S} ===" | tee -a "$OUTPUT_FILE"
    
    for STAGE in "${STAGES[@]}"; do
        echo "Testing Stage: $STAGE"
        echo "--- Implementation: $STAGE ---" >> "$OUTPUT_FILE"
        
        # Run the benchmark wrapped in perf. 
        # Using '| tee -a' ensures the C++ program's runtime table prints to the terminal.
        # Note: Change ./bin/matmul to ./matmul if your binary is built in the root directory.
        perf stat -e instructions,L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses -o perf_tmp.txt \
            ./bin/matmul $STAGE $S $S $S | tee -a "$OUTPUT_FILE"
        
        # Parse standard perf output formatting
        INSTRUCTIONS=$(awk '/instructions/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
        L1_MISSES=$(awk '/L1-dcache-load-misses/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
        L2_MISSES=$(awk '/l2_rqsts.miss/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
        LLC_MISSES=$(awk '/LLC-load-misses/ {gsub(/,/, "", $1); print $1}' perf_tmp.txt)
        ELAPSED_TIME=$(awk '/seconds time elapsed/ {print $1}' perf_tmp.txt)
        
        # Handle unsupported hardware counters gracefully
        [ -z "$L2_MISSES" ] || [[ "$L2_MISSES" == *"<not"* ]] && L2_MISSES="N/A (Check arch spec)"
        [ -z "$LLC_MISSES" ] || [[ "$LLC_MISSES" == *"<not"* ]] && LLC_MISSES="N/A"

        # Calculate Misses Per Kilo-Instruction (MPKI) for L1D
        if [ ! -z "$INSTRUCTIONS" ] && [ "$INSTRUCTIONS" -gt 0 ] && [ ! -z "$L1_MISSES" ] && [[ "$L1_MISSES" != *"<not"* ]]; then
            L1_MPKI=$(echo "scale=4; ($L1_MISSES / ($INSTRUCTIONS / 1000))" | bc)
        else
            L1_MPKI="N/A"
        fi
        
        # Print explicitly
        echo "-> Perf Time:    $ELAPSED_TIME seconds" | tee -a "$OUTPUT_FILE"
        echo "-> Instructions: $INSTRUCTIONS" | tee -a "$OUTPUT_FILE"
        echo "-> L1D Misses:   $L1_MISSES (MPKI: $L1_MPKI)" | tee -a "$OUTPUT_FILE"
        echo "-> L2 Misses:    $L2_MISSES" | tee -a "$OUTPUT_FILE"
        echo "-> LLC Misses:   $LLC_MISSES" | tee -a "$OUTPUT_FILE"
        echo "----------------------------------------" >> "$OUTPUT_FILE"
    done
    echo "" | tee -a "$OUTPUT_FILE"
done

# Clean up temporary perf file
rm -f perf_tmp.txt

echo "========================================================"
echo "Sweep complete. Detailed cache miss results saved to $OUTPUT_FILE."