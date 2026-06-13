#!/bin/bash

# --- 1. SET GLOBAL CONFIGURATIONS ---
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

MODEL_PATH=./model_ckpts/HunyuanVideo-1.5
AR_ACTION_MODEL_PATH=./.trainOutput/rubiksCube/checkpoint-960/transformer/diffusion_pytorch_model.safetensors
# AR_ACTION_MODEL_PATH=./model_ckpts/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors 
BASE_OUTPUT_PATH=./outputs/batch_results
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# Inference Params
PROMPT='A paved pathway leads towards a stone arch bridge...'
IMAGE_PATH=./assets/img/test.png 
SEED=2
ASPECT_RATIO=16:9
RESOLUTION=480p
NUM_FRAMES=125
WIDTH=832
HEIGHT=480
N_INFERENCE_GPU=4
REWRITE=false
ENABLE_SR=false

# --- 2. DEFINE THE LIST OF JSON FILES ---
# Option A: Explicit list
# JSON_FILES=("../path/to/data1.json" "../path/to/data2.json")

# Option B: Find all data.json files in a specific parent directory (Recommended)
JSON_FILES=($(find ../.debugRubiksCubeBase -name "data.json"))

echo "Found ${#JSON_FILES[@]} JSON files to process."

# --- 3. LOOP THROUGH FILES ---
for i in "${!JSON_FILES[@]}"; do
    CURRENT_JSON="${JSON_FILES[$i]}"
    
    # Extract a unique name for the output folder based on the parent directory of the JSON
    # e.g., if path is .../20260514_065822/data.json, FOLDER_NAME is 20260514_065822
    FOLDER_NAME=$(basename $(dirname "$CURRENT_JSON"))
    SPECIFIC_OUTPUT_PATH="$BASE_OUTPUT_PATH/$FOLDER_NAME"
    
    echo "----------------------------------------------------"
    echo "Processing [$((i+1))/${#JSON_FILES[@]}]: $CURRENT_JSON"
    echo "Output Path: $SPECIFIC_OUTPUT_PATH"
    echo "----------------------------------------------------"

    # Prepare arguments for this specific file
    EXTRA_AUTOPLAY_ARGS=(
        --autoplay_data_json "$CURRENT_JSON"
        --autoplay_data_index 0
    )

    # Run Inference
    torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
      --prompt "$PROMPT" \
      --image_path $IMAGE_PATH \
      --resolution $RESOLUTION \
      --aspect_ratio $ASPECT_RATIO \
      --video_length $NUM_FRAMES \
      --seed $SEED \
      --rewrite $REWRITE \
      --sr $ENABLE_SR --save_pre_sr_video \
      --pose "static" \
      "${EXTRA_AUTOPLAY_ARGS[@]}" \
      --output_path "$SPECIFIC_OUTPUT_PATH" \
      --model_path $MODEL_PATH \
      --action_ckpt $AR_ACTION_MODEL_PATH \
      --few_step false \
      --width $WIDTH \
      --height $HEIGHT \
      --model_type 'ar'

    echo "Finished processing $FOLDER_NAME"
done

echo "All tasks complete."