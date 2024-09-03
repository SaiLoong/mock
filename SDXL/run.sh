export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # 据说能更好地会收显存碎片

MODEL_PATH="/mnt/workspace/model/stable-diffusion-xl-base-1.0"
VAE_PATH="/mnt/workspace/model/sdxl-vae-fp16-fix"
DATASET_PATH="/mnt/workspace/dataset/pokemon-blip-captions"
OUTPUT_PATH="/mnt/workspace/LoRA_model/sdxl-pokemon"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --variant="fp16" \
  --dataset_name=$DATASET_PATH \
  --validation_prompt="a black and yellow dragon flying to the moon" \
  --num_validation_images=4 \
  --validation_epochs=4 \
  --output_dir=$OUTPUT_PATH \
  --seed=1024 \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --dataloader_num_workers=8 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --report_to="wandb"
