#!/bin/bash

#SBATCH --exclusive

# Loading the required module
source /etc/profile
module load julia

# Declare output file
out_file="comparison_log.txt"
# Create result file
touch results/${out_file}
# with reading, writing and executing permissions - why is executing needed?
chmod +rwx results/${out_file}

# run code while redirecting stderr to stdout with 2>&1 (https://stackoverflow.com/questions/818255/what-does-21-mean)
julia comparison_test.jl >&1 | tee results/${out_file}