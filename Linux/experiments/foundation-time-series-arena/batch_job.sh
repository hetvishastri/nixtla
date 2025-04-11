#!/bin/bash
#SBATCH --mem=48G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH --gres=gpu:a16:1 # Number and type of GPUs
#SBATCH -t 02:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load conda/latest
conda activate foundation-ts

python -m xiuhmolpilli.arena amazon/chronos-t5-tiny chronos_tiny &
PYTHON_PID=$!
nvidia-smi --query-gpu=timestamp,index,name,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv --loop-ms=100 > usage_tiny.csv &
NVIDIA_SMI_PID=$!
wait $PYTHON_PID
kill $NVIDIA_SMI_PID

python -m xiuhmolpilli.arena amazon/chronos-t5-mini chronos_mini &
PYTHON_PID=$!
nvidia-smi --query-gpu=timestamp,index,name,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv --loop-ms=100 > usage_mini.csv &
NVIDIA_SMI_PID=$!
wait $PYTHON_PID
kill $NVIDIA_SMI_PID

python -m xiuhmolpilli.arena amazon/chronos-t5-small chronos_small &
PYTHON_PID=$!
nvidia-smi --query-gpu=timestamp,index,name,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv --loop-ms=100 > usage_small.csv &
NVIDIA_SMI_PID=$!
wait $PYTHON_PID
kill $NVIDIA_SMI_PID

python -m xiuhmolpilli.arena amazon/chronos-t5-base chronos_base &
PYTHON_PID=$!
nvidia-smi --query-gpu=timestamp,index,name,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv --loop-ms=100 > usage_base.csv &
NVIDIA_SMI_PID=$!
wait $PYTHON_PID
kill $NVIDIA_SMI_PID

python -m xiuhmolpilli.arena amazon/chronos-t5-large chronos_large &
PYTHON_PID=$!
nvidia-smi --query-gpu=timestamp,index,name,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv --loop-ms=100 > usage_large.csv &
NVIDIA_SMI_PID=$!
wait $PYTHON_PID
kill $NVIDIA_SMI_PID
