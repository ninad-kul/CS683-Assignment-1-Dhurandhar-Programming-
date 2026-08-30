#This is a shell script i have used to evaluate the MPKI metric automatically for all the implementations of 2D conv

#!/bin/bash

echo "Building project..."
make
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi
echo -e "Build successful.\n"

# Target list for which codes to evaluate
IMPLEMENTATIONS=("naive" "reorder" "unroll" "tile" "simd" "optimized")

# using hardware cache events (I am using WSL2)
METRICS="instructions,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores"

# print table header
printf "%-12s | %-14s | %-12s | %-12s | %-12s | %-8s\n" "Version" "Instructions" "L1 Loads" "L1 Misses" "L1 Stores" "MPKI"
printf "%s\n" "__________________________________________________________________________________________________________"

for impl in "${IMPLEMENTATIONS[@]}"; do

    OUTPUT=$(sudo perf stat -x , -e $METRICS ./bin/conv $impl 2>&1)
    
    # extract values using awk
    INSTR=$(echo "$OUTPUT" | awk -F, '/instructions/ {print $1}')
    LOADS=$(echo "$OUTPUT" | awk -F, '/L1-dcache-loads/ {print $1}')
    MISSES=$(echo "$OUTPUT" | awk -F, '/L1-dcache-load-misses/ {print $1}')
    STORES=$(echo "$OUTPUT" | awk -F, '/L1-dcache-stores/ {print $1}')
    
    # calculate MPKI = (Misses / Instructions) * 1000
    if [[ -n "$INSTR" && "$INSTR" -gt 0 && -n "$MISSES" && "$MISSES" != "<not supported>" ]]; then
        MPKI=$(awk "BEGIN {printf \"%.2f\", ($MISSES / $INSTR) * 1000}")
    else
        MPKI="N/A"
    fi

    # print formatted row
    printf "%-12s | %-14s | %-12s | %-12s | %-12s | %-8s\n" "$impl" "${INSTR:-N/A}" "${LOADS:-N/A}" "${MISSES:-N/A}" "${STORES:-N/A}" "$MPKI"
done