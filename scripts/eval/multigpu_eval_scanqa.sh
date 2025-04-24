# !/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_scanqa \
        --scan-folder ./playground/data/scannet/val \
        --model-path djiajunustc/3D-LLaVA-7B-LoRA \
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file ./playground/data/eval_info/scanqa/scanqa_val_question.jsonl \
        --answers-file ./playground/predictions/$EXP_NAME/scanqa/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/predictions/$EXP_NAME/scanqa/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/predictions/$EXP_NAME/scanqa/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_scanqa.py \
    --annotation-file ./playground/data/eval_info/scanqa/scanqa_val_answer.jsonl \
    --result-file $output_file
 