#!/bin/bash
#SBATCH --error=job_error_file.txt

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename> <mode>"
    echo "mode: contrastive or supervised"
    exit 1
fi

# Read the filename and mode from the command line arguments
filename=$1
mode=$2

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "Error: File not found: $filename"
    exit 1
fi

# Count the number of lines in the file
num_lines=$(wc -l < "$filename")

# Submit the job array to Slurm
sbatch --array=0-$((num_lines-1)) job_script.sh "$filename" "$mode"
