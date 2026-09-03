#!/bin/bash

echo "Building project..."
make clean > /dev/null 2>&1
make
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi
echo "Build successful."
echo ""

printf "%-12s | %-14s | %-12s | %-12s | %-12s | %-6s\n" "Version" "Instructions" "L1 Loads" "L1 Misses" "L1 Stores" "MPKI"
echo "__________________________________________________________________________________________________________"

STAGES=("naive" "simd" "prefetch" "optimized")

for stage in "${STAGES[@]}"; do
    # Run perf stat capturing comma separated event output
    perf_out=$(sudo perf stat -x, -e instructions,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores ./bin/matmul "$stage" 2>&1)

    # Extract event counts
    inst=$(echo "$perf_out" | grep "instructions" | cut -d',' -f1 | tr -d ' ')
    loads=$(echo "$perf_out" | grep "L1-dcache-loads" | cut -d',' -f1 | tr -d ' ')
    misses=$(echo "$perf_out" | grep "L1-dcache-load-misses" | cut -d',' -f1 | tr -d ' ')
    stores=$(echo "$perf_out" | grep "L1-dcache-stores" | cut -d',' -f1 | tr -d ' ')

    # default values
    inst=${inst:-0}
    loads=${loads:-0}
    misses=${misses:-0}
    stores=${stores:-0}

    # Calculate MPKI = (L1 Load Misses * 1000) / Total Instructions
    mpki=$(awk -v m="$misses" -v i="$inst" 'BEGIN { if (i > 0) printf "%.2f", (m * 1000.0) / i; else print "0.00" }')

    printf "%-12s | %-14s | %-12s | %-12s | %-12s | %-6s\n" "$stage" "$inst" "$loads" "$misses" "$stores" "$mpki"
done
