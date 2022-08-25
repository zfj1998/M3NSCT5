#!/bin/bash

#SBATCH --account=nobody
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=240:00:00
#SBATCH --job-name=ct5
#SBATCH --output=log.ct5_200

module unload nvidia/cuda/10.0
module load nvidia/cuda/10.2

cd $SLURM_SUBMIT_DIR

python main.py \
    --model_name_or_path "./out/ct5/checkpoint-XX" \
    --dataset_cache_path "./dataset/no_tags.8_langs" \
    --output_dir "./out/ct5" \
    --predict_config_file "./configs/ct5_sample200.json" \
    --source_prefix 'summarize: ' \
    --do_predict \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --text_column "code_tokens" \
    --summary_column "docstring_tokens" \
    --max_source_length 512 \
    --max_target_length 32 \
    --do_sample \
