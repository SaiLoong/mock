export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # 减少显存碎片

MODEL_PATH="/mnt/workspace/model/awportrait_v14"
DATASET_PATH="/mnt/workspace/dataset/yuer"
OUTPUT_PATH="/mnt/workspace/LoRA_model/awportrait_v14-yuer"

# 8330_17.jpg的prompt，aw效果不错
PROMPT="1girl, young woman, Asian, solo, outdoor, Yuer, park, portrait, 20-year-old, forest, green trees, sunlight, white dress, straw hat, long hair, gentle expression, natural light, nature, summer, day time, casual, serene, peaceful, relaxed, light makeup"
# TODO 尝试snr_gamma=5

# TODO validation_epochs=4、


python yuer_train_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir=$DATASET_PATH \
  --validation_prompt="$PROMPT" \
  --num_validation_images=4 \
  --validation_epochs=1 \
  --output_dir=$OUTPUT_PATH \
  --seed=1024 \
  --random_flip \
  --train_batch_size=8 \
  --num_train_epochs=30 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=0 \
  --lr_num_cycles=2 \
  --dataloader_num_workers=8 \
  --max_grad_norm=1 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \
  --rank=128

# --max_train_samples=32
