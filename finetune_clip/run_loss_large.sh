#!/bin/bash

set -eux

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)

export CUDA_VISIBLE_DEVICES=$1

train_batch_size=16
eval_batch_size=16

run_finetune() {
    python finetune_clip.py \
      --output_dir ../output/"$1" \
      --model_name_or_path "$2" \
      --train_file ../data/"$3" \
      --validation_file ../data/"$4" \
      --image_column image \
      --overwrite_output_dir=True \
      --max_seq_length=77 \
      --num_train_epochs="$5" \
      --caption_column caption \
      --remove_unused_columns=False \
      --bf16 \
      --do_train \
      --do_eval \
      --evaluation_strategy=epoch \
      --eval_steps=1 \
      --load_best_model_at_end=True \
      --metric_for_best_model=eval_loss \
      --save_strategy=epoch \
      --per_device_train_batch_size="${train_batch_size}" \
      --per_device_eval_batch_size="${eval_batch_size}" \
      --gradient_accumulation_steps=4 \
      --eval_accumulation_steps=16 \
      --weight_decay=0.1 \
      --learning_rate="$6" \
      --warmup_steps=50 \
      --report_to "none" \
      --margin_loss True \
      --margin_loss_weight "$7" \
      --ne_margin_loss True \
      --ne_margin_loss_weight "$8"

      cd ../evaluate_clip

      python main_aro.py --model-name "hf-clip:../output/""$1" --dataset "COCO_Object"
      python main_aro.py --model-name "hf-clip:../output/""$1" --dataset "Flickr_Object"
      python main_aro.py --model-name "hf-clip:../output/""$1" --dataset "Nocaps_Object"
      python main_aro.py --model-name "hf-clip:../output/""$1" --dataset "VG_Relation"
      python main_aro.py --model-name "hf-clip:../output/""$1" --dataset "VG_Attribution"

      cd ../finetune_clip
}

run_finetune "clip_finetune_mix_contrastive_1e-5_3_margin_all_0.2_0.2" "../cache/model/clip-vit-large-patch14-336" \
"IHD/train/all_train.json" "IHD/train/dev_data.json" "3" "1e-5" "0.2" "0.2"