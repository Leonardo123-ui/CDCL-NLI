#!/bin/bash

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "Usage: $0 <python_script> <output_file>"
    echo "Example: $0 train.py output.log"
    exit 1
fi

# 获取参数
SCRIPT=$1
OUTPUT=$2

# 检查 Python 脚本是否存在
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found!"
    exit 1
fi

# 记录开始信息
{
    echo "Command: nohup python -u $SCRIPT"
    echo "Start time: $(date)"
    echo "Host: $(hostname)"
    echo "Working directory: $(pwd)"
    echo "GPU info: $(nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader)"
    echo "-------------------"
} > "$OUTPUT"

# 运行主程序
nohup python -u "$SCRIPT" >> "$OUTPUT" 2>&1 &
PID=$!

# 记录PID
echo "Process ID: $PID" >> "$OUTPUT"
echo "Started process $PID, output redirected to $OUTPUT"
