export MODEL_NAME="/mnt/workspace/model/stable-diffusion-v1-5"
export OUTPUT_DIR="/mnt/workspace/LoRA_model/sd-v1-5-pokemon"
export DATASET_NAME="/mnt/workspace/dataset/pokemon-blip-captions"

python train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --mixed_precision="fp16" \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="a cartoon dragon" \
  --seed=1024 \
  --validation_epochs 1
