# https://huggingface.co/docs/transformers/main_classes/trainer

python run_summarization.py \
    --model_name_or_path north/t5_base_NCC_lm \
    --do_train \
    --do_predict \
    --do_eval \
    --dataset_name navjordj/SNL_summarization \
    --text_column article \
    --summary_column ingress \
    --source_prefix "oppsummer: " \
    --output_dir snl-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --num_train_epochs 1 \
    --predict_with_generate \
    --save_total_limit 2 \
    --save_steps 100 \
    --logging_steps 100 \
    --max_source_length 512
