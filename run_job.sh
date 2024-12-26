#!/bin/bash

# check number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <python_script> <output_file>"
    echo "Example: $0 train.py output.log"
    exit 1
fi

# get arguments
SCRIPT=$1
OUTPUT=$2

# check if python script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found!"
    exit 1
fi

# start the job
{
    echo "Command: nohup python -u $SCRIPT"
    echo "Start time: $(date)"
    echo "Host: $(hostname)"
    echo "Working directory: $(pwd)"
    echo "GPU info: $(nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader)"
    echo "-------------------"
} > "$OUTPUT"

# run the python script in the background
nohup python -u "$SCRIPT" >> "$OUTPUT" 2>&1 &
PID=$!

# record the process ID
echo "Process ID: $PID" >> "$OUTPUT"
echo "Started process $PID, output redirected to $OUTPUT"
