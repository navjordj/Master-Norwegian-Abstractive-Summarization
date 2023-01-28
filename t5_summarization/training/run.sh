# https://huggingface.co/docs/transformers/main_classes/trainer

python run_summarization.py \
    --model_name_or_path north/t5_small_NCC_lm \
    --do_train \
    --train_file vg_summarization.csv \
    --text_column article \
    --summary_column ingress \
    --source_prefix "oppsummer: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --predict_with_generate \
    --save_total_limit 2 \
    --save_steps 5000 \
    --num_train_epochs 2