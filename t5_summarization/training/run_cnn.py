import os

command = """
accelerate launch --config_file accelerate_config.yaml run_summarization.py \
    --model_name_or_path north/t5_base_NCC_lm \
    --do_train \
    --do_predict \
    --do_eval \
    --dataset_name jkorsvik/cnn_daily_mail_nor_final \
    --text_column article \
    --summary_column highlights \
    --source_prefix "oppsummer: " \
    --output_dir t5-base-cnndaily \
    --overwrite_output_dir \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs 20 \
    --predict_with_generate \
    --save_total_limit 2 \
    --logging_steps 25 \
    --max_source_length 512 \
    --gradient_accumulation_steps 4 \
    --push_to_hub \
    --save_strategy epoch \
    --load_best_model_at_end \
    --evaluation_strategy epoch \
    --ddp_find_unused_parameters False \
    --metric_for_best_model eval_loss \
    --generation_max_length 80
"""

os.system(command)

