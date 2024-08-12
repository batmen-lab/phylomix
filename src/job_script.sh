#!/bin/bash

#SBATCH --job-name=myJob
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=slurm_logs/myJob_%A_%a.out

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <args_filename> <mode>"
    echo "mode: contrastive or supervised"
    exit 1
fi

# Read the filename and mode from the command line arguments
args_file=$1
mode=$2

# Check if the file exists
if [ ! -f "$args_file" ]; then
    echo "Error: File not found: $args_file"
    exit 1
fi

# Load the arguments for the current task
IFS=$'\n' read -d '' -r -a lines < "$args_file"
args="${lines[$SLURM_ARRAY_TASK_ID]}"

# Determine which script to run based on the mode
if [ "$mode" == "contrastive" ]; then
    python contrastive_learning.py $args
elif [ "$mode" == "supervised" ]; then
    python train.py $args
else
    echo "Error: Invalid mode specified: $mode"
    exit 1
fi

echo "done"
exit 0
