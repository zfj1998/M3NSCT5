#!/bin/bash

#SBATCH --account=nobody
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=240:00:00
#SBATCH --job-name=ct5
#SBATCH --output=log.ct5

module unload nvidia/cuda/10.0
module load nvidia/cuda/10.2

cd $SLURM_SUBMIT_DIR

python main.py \
    --model_name_or_path "./dataset/codet5-base" \
    --dataset_cache_path "./dataset" \
    --output_dir "./out/ct5" \
    --source_prefix 'summarize: ' \
    --do_train \
    --early_stopping_patience 5 \
    --load_best_model_at_end \
    --ignore_data_skip \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --logging_steps 200 \
    --eval_steps 1290 \
    --save_steps 1290 \
    --predict_with_generate \
    --save_total_limit 3 \
    --num_train_epochs 10 \
    --text_column "code_tokens" \
    --summary_column "docstring_tokens" \
    --max_source_length 512 \
    --max_target_length 32 \
    --num_beams 4 \