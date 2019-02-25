#!/usr/bin/env bash
export GLUE_DIR=../glue_data/

python3 run_classifier.py \
  --task_name JAIT \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/JAIT/ \
  --bert_model bert-base-multilingual-cased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/