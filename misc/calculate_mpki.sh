#!/bin/bash

echo "Building project..."
make
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi
echo -e "Build successful.\n"

IMPLEMENTATIONS=("naive" "reorder" "unroll" "tile" "simd" "optimized")

# Removed L1-dcache-stores to prevent Intel PMU crashes
METRICS="instructions,L1-dcache-loads,L1-dcache-load-misses"

printf "%-12s | %-14s | %-12s | %-12s | %-8s\n" "Version" "Instructions" "L1 Loads" "L1 Misses" "MPKI"
printf "%s\n" "_________________________________________________________________________"

for impl in "${IMPLEMENTATIONS[@]}"; do
    # Run perf and output as CSV (-x ,)
    OUTPUT=$(sudo perf stat -x , -e $METRICS ./bin/conv $impl 2>&1)
    
    # Extract only the first column value for the specific events
    INSTR=$(echo "$OUTPUT" | grep -i ",instructions" | awk -F, '{print $1}')
    LOADS=$(echo "$OUTPUT" | grep -i ",L1-dcache-loads" | awk -F, '{print $1}')
    MISSES=$(echo "$OUTPUT" | grep -i ",L1-dcache-load-misses" | awk -F, '{print $1}')
    
    # Calculate MPKI if metrics are valid integers
    if [[ -n "$INSTR" && "$INSTR" -gt 0 && -n "$MISSES" && "$MISSES" =~ ^[0-9]+$ ]]; then
        MPKI=$(awk "BEGIN {printf \"%.2f\", ($MISSES / $INSTR) * 1000}")
    else
        MPKI="N/A"
    fi

    printf "%-12s | %-14s | %-12s | %-12s | %-8s\n" "$impl" "${INSTR:-N/A}" "${LOADS:-N/A}" "${MISSES:-N/A}" "$MPKI"
done