# https://huggingface.co/docs/transformers/main_classes/trainer

python run_summarization.py \
    --model_name_or_path north/t5_large_NCC_lm \
    --do_train \
    --do_predict \
    --dataset_name navjordj/SNL_summarization \
    --text_column article \
    --summary_column ingress \
    --source_prefix "oppsummer: " \
    --output_dir snl-summarization \
    --overwrite_output_dir False \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=8 \
    --predict_with_generate \
    --save_total_limit 2 \
    --save_steps 100 \
    --num_train_epochs 20 \
    --push_to_hub \
    --logging_steps 100 \
    --max_source_length 512