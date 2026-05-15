export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'

IMAGE_PATH=./assets/img/test.png # Now we only provide the i2v model, so the path cannot be None
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
OUTPUT_PATH=./outputs/
MODEL_PATH=./model_ckpts/HunyuanVideo-1.5                   # Path to pretrained hunyuanvideo-1.5 model
# AR_ACTION_MODEL_PATH=./model_ckpts/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors         # Path to our HY-World 1.5 autoregressive checkpoints
AR_ACTION_MODEL_PATH=./.trainOutput2/checkpoint-448/transformer/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=         # Path to our HY-World 1.5 bidirectional checkpoints
AR_DISTILL_ACTION_MODEL_PATH= # Path to our HY-World 1.5 autoregressive distilled checkpoints
POSE=                   # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
AUTOPLAY_DATA_JSON=../.debugSingle/20260514_130002_960297_flappy-bird/data.json          # Optional: path to generateAutoplayDataset.py data.json
AUTOPLAY_DATA_INDEX=0         # Dataset item index to use when AUTOPLAY_DATA_JSON is set
NUM_FRAMES=125
WIDTH=832
HEIGHT=480

EXTRA_AUTOPLAY_ARGS=()
if [ -n "$AUTOPLAY_DATA_JSON" ]; then
  POSE=static                  # Use a fixed camera pose when feeding gameplay actions directly
  EXTRA_AUTOPLAY_ARGS+=(--autoplay_data_json "$AUTOPLAY_DATA_JSON")
  EXTRA_AUTOPLAY_ARGS+=(--autoplay_data_index "$AUTOPLAY_DATA_INDEX")
fi

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=4 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

# inference with bidirectional model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $BI_ACTION_MODEL_PATH \
#   --few_step false \
#   --model_type 'bi'

# inference with autoregressive model
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose "$POSE" \
  "${EXTRA_AUTOPLAY_ARGS[@]}" \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_ACTION_MODEL_PATH \
  --few_step false \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'ar'

# inference with autoregressive distilled model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   "${EXTRA_AUTOPLAY_ARGS[@]}" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
#   --few_step true \
#   --num_inference_steps 4 \
#   --model_type 'ar' \
#   --use_vae_parallel false \
#   --use_sageattn false \
#   --use_fp8_gemm false \
