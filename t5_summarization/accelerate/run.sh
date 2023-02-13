accelerate launch no_trainer.py \
    --model_name_or_path north/t5_large_NCC_lm \
    --dataset_name navjordj/SNL_summarization \
    --text_column article \
    --summary_column ingress \
    --source_prefix "oppsummer: " \
    --output_dir tst-summarization \
    --max_source_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12