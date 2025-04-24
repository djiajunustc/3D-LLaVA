# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXP_NAME=finetune-3d-llava-lora

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_sqa3d \
        --scan-folder ./playground/data/scannet/val \
        --model-path djiajunustc/3D-LLaVA-7B-LoRA \
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval_info/sqa3d/sqa3d_test_question.json \
        --answers-file ./playground/predictions/$EXP_NAME/sqa3d/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/predictions/$EXP_NAME/sqa3d/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/predictions/$EXP_NAME/sqa3d/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_sqa3d.py \
    --annotation-file ./playground/data/eval_info/sqa3d/sqa3d_test_answer.json \
    --result-file $output_file
