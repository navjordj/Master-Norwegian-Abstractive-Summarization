python3 run_summarization_no_trainer.py \
    --model_name_or_path north/t5_small_NCC_lm \
    --dataset_name navjordj/SNL_summarization \
    --text_column article \
    --summary_column ingress \
    --source_prefix "oppsummer: " \
    --output_dir ~/tmp/tst-summarization \
    --per


python3 run_summarization.py \
    --model_name_or_path north/t5_small_NCC_lm \
    --dataset_name navjordj/SNL_summarization \
    --do_train \
    --do_eval \
    --source_prefix "oppsummer: " \
    --text_column article \
    --summary_column ingress \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_source_length 512

