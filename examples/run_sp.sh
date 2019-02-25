#!/usr/bin/env bash
export GLUE_DIR=../glue_data/
export PRETRAINED_MODEL_PATH=/home/weicheng.zhu/experiment/bert-japanese/model

python3 run_classifier_sp.py \
  --task_name RITC \
  --do_train \
  --do_eval \
  --data_dir ../glue_data//RITC/ \
  --bert_model=/home/weicheng.zhu/experiment/bert-japanese/model \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/ritc_output_best/
