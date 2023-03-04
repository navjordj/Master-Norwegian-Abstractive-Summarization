# https://huggingface.co/docs/transformers/main_classes/trainer

accelerate launch --config_file accelerate_config.yaml run_summarization.py \
    --model_name_or_path north/t5_base_NCC_lm \
    --do_train \
    --do_predict \
    --do_eval \
    --dataset_name navjordj/SNL_summarization \
    --text_column article \
    --summary_column ingress \
    --source_prefix "oppsummer: " \
    --output_dir t5-base-snl \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --num_train_epochs 20 \
    --predict_with_generate \
    --save_total_limit 2 \
    --save_steps 25 \
    --logging_steps 25 \
    --max_source_length 512 \
    --gradient_accumulation_steps 2 \
    --push_to_hub \
    --save_strategy epoch \
    --early_stopping_patience 2