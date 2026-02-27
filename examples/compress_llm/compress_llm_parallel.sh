#!/bin/bash
#
# Parallel DFloat11 compression for LLMs.
# Each CPU core compresses a range of decoder layers independently.
#
# Usage:
#   bash compress_llm_parallel.sh <model_name_or_path> <save_path> <num_layers> [blocks_per_task]
#
# Examples:
#   bash compress_llm_parallel.sh Qwen/Qwen2.5-7B  ./Qwen2.5-7B-DF11  28
#   bash compress_llm_parallel.sh meta-llama/Llama-3.1-8B ./Llama-3.1-8B-DF11 32
#   bash compress_llm_parallel.sh Qwen/Qwen3-8B ./Qwen3-8B-DF11 36 6

# MODEL=${1:?"Usage: $0 <model_name_or_path> <save_path> <num_layers> [blocks_per_task]"}
# SAVE_PATH=${2:?"Usage: $0 <model_name_or_path> <save_path> <num_layers> [blocks_per_task]"}
# NUM_LAYERS=${3:?"Usage: $0 <model_name_or_path> <save_path> <num_layers> [blocks_per_task]"}
# BLOCKS_PER_TASK=${4:-4}

MODEL=Qwen/Qwen3-1.7B
SAVE_PATH=./Qwen3-1.7B-DF11
NUM_LAYERS=28
BLOCKS_PER_TASK=4


CORE=0
for ((i=0; i<NUM_LAYERS; i+=BLOCKS_PER_TASK))
do
    END=$((i+BLOCKS_PER_TASK))
    echo "Compressing layers ${i} to ${END} on CPU core ${CORE}"
    taskset -c ${CORE} python compress_llm.py \
        --model_name_or_path ${MODEL} \
        --save_path ${SAVE_PATH} \
        --check_correctness \
        --block_range ${i} ${END} &
    CORE=$((CORE+1))
done

echo "Launched ${CORE} parallel compression tasks. Waiting..."
wait
echo "All compression tasks finished."
