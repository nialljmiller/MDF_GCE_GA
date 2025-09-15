#!/bin/bash
# summarize_seff.sh
# Summarize CPU, memory, walltime, and efficiency across all jobs in current dir

total_cpu=0
total_wall=0
total_mem=0
max_mem=0
jobs=0
eff_sum=0

for f in *.out; do
    jobid="${f##*_}"
    jobid="${jobid%.out}"
    stats=$(seff "$jobid" 2>/dev/null)

    cpu=$(echo "$stats" | awk -F': ' '/CPU Utilized/ {print $2}')
    wall=$(echo "$stats" | awk -F': ' '/Job Wall-clock time/ {print $2}')
    mem=$(echo "$stats" | awk -F': ' '/Memory Utilized/ {print $2}' | awk '{print $1}')
    eff=$(echo "$stats" | awk -F': ' '/CPU Efficiency/ {print $2}' | tr -d '%')

    # convert CPU time and wall time (D-HH:MM:SS → seconds)
    cpu_secs=$(echo "$cpu" | awk -F'[-:]' '{if (NF==3) {d=0; h=$1; m=$2; s=$3} else {d=$1; h=$2; m=$3; s=$4} print (d*86400)+(h*3600)+(m*60)+s}')
    wall_secs=$(echo "$wall" | awk -F'[-:]' '{if (NF==3) {d=0; h=$1; m=$2; s=$3} else {d=$1; h=$2; m=$3; s=$4} print (d*86400)+(h*3600)+(m*60)+s}')

    total_cpu=$((total_cpu + cpu_secs))
    total_wall=$((total_wall + wall_secs))
    total_mem=$((total_mem + mem))
    (( mem > max_mem )) && max_mem=$mem
    eff_sum=$(echo "$eff_sum + $eff" | bc -l)

    ((jobs++))
done

avg_cpu_eff=$(echo "scale=2; $eff_sum/$jobs" | bc -l)
avg_mem=$(echo "scale=2; $total_mem/$jobs" | bc -l)

echo "Jobs analyzed: $jobs"
echo "Total CPU Utilized: $(date -u -d @$total_cpu +%T) (=$total_cpu sec)"
echo "Total Wall-clock: $(date -u -d @$total_wall +%T) (=$total_wall sec)"
echo "Average CPU Efficiency: $avg_cpu_eff %"
echo "Average Memory Utilized: $avg_mem MB"
echo "Max Memory Utilized: $max_mem MB"

