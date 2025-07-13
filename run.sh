#!/bin/bash
# 模型訓練
python train.py \
  --train_csv dataset/train/train.csv \
  --val_csv dataset/val/val.csv \
  --output_dir wav2vec2-finetuned-opdir \
  --train_batch_size 8 \
  --eval_batch_size 2 \
  --num_train_epochs 5 \
  --save_strategy steps \
  --save_steps 1000 \
  --eval_strategy steps \
  --eval_steps 1000 \
  --logging_steps 100 \
  --learning_rate 1e-4 \
  --weight_decay 0.005 \
  --warmup_steps 1000 \
  --save_total_limit 15 \
  --fp16 True\
  --gradient_accumulation_steps 2 \
  --vocab_file vocab.json \
  --pretrained_model facebook/wav2vec2-large-xlsr-53

# 使用checkpoint接續訓練
# python continue_train.py \
#   --train_csv dataset/train/train.csv \
#   --val_csv dataset/val/val.csv \
#   --output_dir wav2vec2-finetuned-opdir-continue \
#   --train_batch_size 8 \
#   --eval_batch_size 2 \
#   --num_train_epochs 5 \
#   --save_strategy steps \
#   --save_steps 1000 \
#   --eval_strategy steps \
#   --eval_steps 1000 \
#   --logging_steps 100 \
#   --learning_rate 1e-4 \
#   --weight_decay 0.005 \
#   --warmup_steps 1000 \
#   --save_total_limit 2 \
#   --fp16 True\
#   --gradient_accumulation_steps 2 \
#   --vocab_file vocab.json \
#   --pretrained_model facebook/wav2vec2-large-xlsr-53 \
#   --checkpoint_path wav2vec2-finetuned-opdir/<checkpoint-dir>

# 驗證測試資料
python eval.py \
  --checkpoint_path_or_repo_id wav2vec2-finetuned-opdir \
  --test_data_path dataset/val/val.csv \
  --output_path pred/predictions.txt

# 推送模型到 Hugging Face Hub
# python push_to_huggingface.py \
#   --model_dir wav2vec2-finetuned-opdir \
#   --repo_id <huggingface_ID/model_name> \
#   --hf_token <your_huggingface_token> 


