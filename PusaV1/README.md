# Pusa-Video V1.0

By fine-tuning the state-of-the-art Wan2.1-T2V-14B model with Vectorized Timestep Adaptation (VTA), Pusa-V1.0 unlocks zero-shot capabilities for tasks like image-to-video (I2V), video extension, start-end frames, et al., all while preserving the base model's text-to-video capability. Particularly, Pusa-V1.0 achieves better performance than Wan-I2V in I2V generation with unprecedented efficiency.

## Installation

Before using this model, you may follow the code below to setup the environment, Cuda 12.4 recommended.
```shell
conda create -n pusav1 python=3.10 -y
conda activate pusav1
cd ./PusaV1
pip install -e .
pip install xfuser>=0.4.3 absl-py peft lightning pandas deepspeed wandb av 
```

## Model Preparation

Download the necessary models and place them into the `PusaV1/model_zoo` directory. You can use the following commands to download and arrange the models correctly.

```shell
# Make sure you are in the PusaV1 directory
# Install huggingface-cli if you don't have it
pip install -U "huggingface_hub[cli]"
huggingface-cli download RaphaelLiu/PusaV1 --local-dir ./PusaV1/model_zoo/
cat ./PusaV1/pusa_v1.pt.part* > ./PusaV1/pusa_v1.pt
```

## Usage Examples

All scripts save their output in an `outputs` directory, which will be created if it doesn't exist.

### Image-to-Video Generation

This script generates a video conditioned on an input image and a text prompt.

```shell
python examples/pusavideo/wan_14b_image_to_video_pusa.py \
  --image_path "./demos/input_image.jpg" \
  --prompt "A wide-angle shot shows a serene monk meditating perched a top of the letter E of a pile of weathered rocks that vertically spell out 'ZEN'. The rock formation is perched atop a misty mountain peak at sunrise. The warm light bathes the monk in a gentle glow, highlighting the folds of his saffron robes. The sky behind him is a soft gradient of pink and orange, creating a tranquil backdrop. The camera slowly zooms in, capturing the monk's peaceful expression and the intricate details of the rocks. The scene is bathed in a soft, ethereal light, emphasizing the spiritual atmosphere." \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt" \
```

### Video-to-Video Generation

This script can be used for various video-to-video tasks like video completion, video extension, or video transition, by providing an input video with at least 81 frames and specify condition settings. The generated video has 81 frames/21 latent frames in total.

**Example 1: Video Completion (Start-End Frames)**
Give the start frame and 4 end frames (encoded to one single latent frame) as conditions. 

```shell
python examples/pusavideo/wan_14b_v2v_pusa.py \
  --video_path "./demos/input_video.mp4" \
  --prompt "piggy bank surfing a tube in teahupo'o wave dusk light cinematic shot shot in 35mm film" \
  --cond_position "0,20" \
  --noise_multipliers "0,0" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt"
```

**Example 2: Video Extension**
Give 13 frames as condition (encoded to the first 4 latent frames). 

```shell
python examples/pusavideo/wan_14b_v2v_pusa.py \
  --video_path "./demos/input_video.mp4" \
  --prompt "piggy bank surfing a tube in teahupo'o wave dusk light cinematic shot shot in 35mm film" \
  --cond_position "0,1,2,3" \
  --noise_multipliers "0,0,0,0" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt"
```

### Multi-Frame Conditioned Generation

This script generates a video conditioned on multiple input frames and a prompt.

**Example: Start-End Frames**
Give the start and end frames as image files for conditioning, and add some noise to the condition frames to generate more coherent video.

```shell
python examples/pusavideo/wan_14b_multi_frames_pusa.py \
  --image_paths "./demos/start_frame.jpg" "./demos/end_frame.jpg" \
  --prompt "plastic injection machine opens releasing a soft inflatable foamy morphing sticky figure over a hand. isometric. low light. dramatic light. macro shot. real footage" \
  --cond_position "0,20" \
  --noise_multipliers "0.3,0.7" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt"
```

### Text-to-Video Generation

This script generates a video from a text prompt.

```shell
python examples/pusavideo/wan_14b_text_to_video_pusa.py \
  --prompt "A vibrant coral reef teeming with life, schools of colorful fish darting through the intricate coral formations. A majestic sea turtle glides gracefully past, its shell a mosaic of earthy tones. Sunlight filters through the clear blue water, creating a breathtaking underwater spectacle." \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt"
```

## Training
Our training pipeline is based on Diffsynth-Studio, which supports both full finetuing and lora finetuing. We use LoRA training on a custom dataset to get Pusa V1.0 model. The training process consists of two stages: data preparation and training. 

### Prepare Dataset
You can download our dataset on Huggingface or prepare our own dataset following https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo.

Download `PusaV1_training` dataset to here `./dataset/`.
```shell
huggingface-cli download RaphaelLiu/PusaV1_training --repo-type dataset --local-dir ./dataset/
```

### Training
After prepraring the dataset, you can start training. We provide a sample script `train.sh` for multi-GPU training on a single node using `torchrun` and `deepspeed`.

You can find the content in `examples/pusavideo/train.sh` and modify the paths and parameters as needed. Finally, run the script from the `PusaV1` directory:
```shell
bash ./examples/pusavideo/train.sh
```
The trained LoRA model will be saved in the `lightning_logs` directory inside your specified `--output_path`.



