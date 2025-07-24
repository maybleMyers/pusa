# Pusa V1.0

<p align="center">
    <a href="https://yaofang-liu.github.io/Pusa_Web/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge"></a>
    <a href="https://arxiv.org/abs/2507.16116"><img alt="Technical Report" src="https://img.shields.io/badge/Technical_Report-📜-B31B1B?style=for-the-badge"></a>
    <a href="https://huggingface.co/RaphaelLiu/PusaV1"><img alt="Model" src="https://img.shields.io/badge/Pusa_V1.0-Model-FFD700?style=for-the-badge&logo=huggingface"></a>
    <a href="https://huggingface.co/datasets/RaphaelLiu/PusaV1_training"><img alt="Dataset" src="https://img.shields.io/badge/Pusa_V1.0-Dataset-6495ED?style=for-the-badge&logo=huggingface"></a>
</p>
<p align="center">
    <a href="https://github.com/Yaofang-Liu/Mochi-Full-Finetuner"><img alt="Code" src="https://img.shields.io/badge/Code-Training%20Scripts-32CD32?logo=github"></a>
    <a href="https://arxiv.org/abs/2410.03160"><img alt="Paper" src="https://img.shields.io/badge/📜-FVDM%20Paper-B31B1B?logo=arxiv"></a>
    <a href="https://x.com/stephenajason"><img alt="Twitter" src="https://img.shields.io/badge/🐦-Twitter-1DA1F2?logo=twitter"></a>
    <a href="https://www.xiaohongshu.com/discovery/item/67f898dc000000001c008339"><img alt="Xiaohongshu" src="https://img.shields.io/badge/📕-Xiaohongshu-FF2442"></a>
</p>

## 📑 Table of Contents

- [🔥 Announcing Pusa V1.0](#-announcing-pusa-v10-)
- [✨ Highlights](#sparkles-highlights)
- [🛠️ Installation](#installation)
- [📦 Model Preparation](#model-preparation)
- [🎬 Gradio Demo (Work in Progress)](#-gradio-demo-work-in-progress)
- [🚀 Usage Examples](#usage-examples)
  - [Image(s) Conditioned Video Generation](#images-conditioned-video-generation)
  - [Video-to-Video Generation](#video-to-video-generation)
  - [Text-to-Video Generation](#text-to-video-generation)
- [🏋️ Training](#training)
  - [Prepare Dataset](#prepare-dataset)
  - [Training Process](#training-1)

---

## 🔥🔥🔥🚀 Announcing Pusa V1.0 🚀🔥🔥🔥

We are excited to release **Pusa V1.0**, a groundbreaking paradigm that leverages **vectorized timestep adaptation (VTA)** to enable fine-grained temporal control within a unified video diffusion framework. By finetuning the SOTA **Wan-T2V-14B** model with VTA, Pusa V1.0 achieves unprecedented efficiency, **surpassing Wan-I2V on Vbench-I2V with only $500 of training cost and 4k data**. The codebase has been integrated into the `PusaV1` directory, based on `DiffSynth-Studio`. Pusa V1.0 not only sets a new standard for image-to-video generation but also unlocks many other zero-shot multi-task capabilities such as start-end frames and video extension, all without task-specific training while preserving the base model's T2V capabilities.

**Important Note!!!**  
**Our method also works for Full Finetuning with extremely low cost (see [Pusa V0.5](https://github.com/Yaofang-Liu/Mochi-Full-Finetuner), only $100 full finetuing cost). The model gains mainly come from our method, not Lora. Lora is insignificant here.**. We use Lora only because, Wan2.1 full finetuing need too many GPUs (at least 32x80G GPUs, yet we only have 24), so that we choose Lora with very large rank to approximate full finetuing. Why 512 rank is just because  its the largest rank that can train with 8x80G GPUs (768 would cause OOM). Besides, we also did full finetuning to the base model with 81 frames in 480p data or 65 frames in 720p data with less GPUs, our method also works. **We suggest more to try full finetuing with our method if you have the resources. We believe the performance could be further imporved!**

## :sparkles: Highlights
- [ComfyUI](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Pusa), supported by [Kijai](https://github.com/kijai), thanks a lot! 

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

Download the necessary models and place them into the `./model_zoo` directory. You can use the following commands to download and arrange the models correctly.

```shell
# Make sure you are in the PusaV1 directory
# Install huggingface-cli if you don't have it
pip install -U "huggingface_hub[cli]"
huggingface-cli download RaphaelLiu/PusaV1 --local-dir ./model_zoo/PusaV1

# (Optional) Please download Wan2.1-T2V-14B to ./model_zoo/PusaV1 is you don't have it, if you have you can directly soft link it to ./model_zoo/PusaV1
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./model_zoo/PusaV1/Wan2.1-T2V-14B
```

The checkpoints should arrange like this to use the codes with default settings:
```shell
./model_zoo
  - PusaV1
    - Wan2.1-T2V-14B
    - pusa_v1.pt
```

## 🎬 Gradio Demo (Work in Progress)

For the easiest way to experience Pusa V1.0, we provide a beautiful web-based Gradio demo with an intuitive interface:


```shell
# Launch the demo
bash launch_demo.sh

# Or run directly
python examples/pusavideo/run_demo.py
```

<div style="display: flex; justify-content: center; gap: 2px;">
  <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/22f96c7de9b1f98bc373505bcc0cb846954dea8f/PusaV1/assets/gradio_page1_1.png?raw=true" 
       alt="Image 1" width="49.5%">
  <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/22f96c7de9b1f98bc373505bcc0cb846954dea8f/PusaV1/assets/gradio_page2_1.png?raw=true" 
       alt="Image 2" width="49.5%">
</div>

The demo will be available at `http://localhost:7860` and includes:
- **🎨 Image-to-Video (I2V)**: Generate videos from single images with motion control
- **🖼️ Multi-Frame Generation**: Create smooth transitions between start-end frames  
- **🎥 Video-to-Video (V2V)**: Extend or complete existing videos
- **📝 Text-to-Video (T2V)**: Generate videos directly from text descriptions
- **📊 Interactive Gallery**: View demo results with exact parameter settings
- **⚙️ Easy Parameter Control**: Adjust LoRA alpha, noise multipliers, and conditioning positions

The Gradio interface provides real-time parameter adjustment and includes pre-configured examples for each generation mode. Perfect for experimentation and getting familiar with Pusa V1.0's capabilities!


## Usage Examples

All scripts save their output in an `outputs` directory, which will be created if it doesn't exist.

!!! :sparkles: **Please note that we have two core unique parameters that differ from other methods. `--cond_position`** is Comma-separated list of frame indices for conditioning. You can use any position from 0 to 20. **`--noise_multipliers`** is "Comma-separated noise multipliers for conditioning frames. A value of 0 means the condition image is used as totally clean, higher value means add more noise. For I2V, you can use 0.2 or any from 0 to 1. For Start-End-Frame, you can use 0.2,0.4, or any from 0 to 1. **`--lora_alpha`** is another very important parameter. A bigger alpha would bring more temporal consistency (i.e., make generated frames more like conditioning part), but may also cause small motion or even collapse. We recommend using a value around 1 to 2. For **`--num_inference_steps`**, **5 steps** are enough to get good results with [LightX2V](https://github.com/ModelTC/LightX2V), which has been intergrated by [Kijai](https://github.com/kijai) in [ComfyUI](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Pusa), check [showcases](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/804#issuecomment-3110144063).

**Try different configurations and you will get different results.** **Examples shown below are just for demonstration and not the best**

### Image(s) Conditioned Video Generation

This script generates a video conditioned on one or more input frames and a text prompt. It can be used for image-to-video, start-end frame conditioned generation, and other multi-frame conditioning tasks.

**Example 1-1: Image-to-Video**
Generates a video from a single starting image. 

```shell
CUDA_VISIBLE_DEVICES=0 python examples/pusavideo/wan_14b_multi_frames_pusa.py \
  --image_paths "./demos/input_image.jpg" \
  --prompt "A wide-angle shot shows a serene monk meditating perched a top of the letter E of a pile of weathered rocks that vertically spell out 'ZEN'. The rock formation is perched atop a misty mountain peak at sunrise. The warm light bathes the monk in a gentle glow, highlighting the folds of his saffron robes. The sky behind him is a soft gradient of pink and orange, creating a tranquil backdrop. The camera slowly zooms in, capturing the monk's peaceful expression and the intricate details of the rocks. The scene is bathed in a soft, ethereal light, emphasizing the spiritual atmosphere." \
  --cond_position "0" \
  --noise_multipliers "0.2" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt" \
  --lora_alpha 1.4 \
  --num_inference_steps 10
```

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_noise_0p0.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.0]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_noise_0p2.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.2]</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_noise_0p3.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.3]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_noise_0p4.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.4]</sub>
    </td>
  </tr>
</table>

**Example 1-2: Image-to-Video with Different LoRA Alpha Values**
Demonstrates the effect of different LoRA alpha values on generation quality and temporal consistency.

```shell
CUDA_VISIBLE_DEVICES=0 python examples/pusavideo/wan_14b_multi_frames_pusa.py \
  --image_paths "./demos/input_image1.jpg" \
  --prompt "A low-angle, long exposure shot of a lone female climber, wearing shorts and tank top rock climbing on a massive asteroid in deep space. The climber is suspended against a star-filled void. dramatic shadows across the asteroid's rugged surface, emphasizing the climber's isolation and the scale of the space rock. Dust particles float in the light beams, catching the light. The climber moves methodically, with focused determination." \
  --cond_position "0" \
  --noise_multipliers "0.2" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt" \
  --lora_alpha 1.4 \
  --num_inference_steps 10
```

<table>
  <tr>
  <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output1_cond_0_noise_0p0_alpha_1p4.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.0], alpha: 1.4</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output1_cond_0_noise_0p0_alpha_1p3.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.0], alpha: 1.3</sub>
    </td>
    
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output1_cond_0_noise_0p0_alpha_1p6.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.0], alpha: 1.6</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output1_cond_0_noise_0p2_alpha_1p3.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.2], alpha: 1.3</sub>
    </td>
  </tr>
</table>

**Example 2: Start-End Frames**
Give the start and end frames as image files for conditioning, and add some noise to the condition frames to generate more coherent video.

```shell
CUDA_VISIBLE_DEVICES=0 python examples/pusavideo/wan_14b_multi_frames_pusa.py \
  --image_paths "./demos/start_frame.jpg" "./demos/end_frame.jpg" \
  --prompt "plastic injection machine opens releasing a soft inflatable foamy morphing sticky figure over a hand. isometric. low light. dramatic light. macro shot. real footage" \
  --cond_position "0,20" \
  --noise_multipliers "0.2,0.5" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt" \
  --lora_alpha 1.4 \
  --num_inference_steps 30
```

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_20_noise_0p1_0p4.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.1, 0.4]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_20_noise_0p2_0p5.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.2, 0.5]</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_20_noise_0p1_0p7.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.1, 0.7]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/multi_frame_output_cond_0_20_noise_0p2_0p6.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.2, 0.6]</sub>
    </td>
  </tr>
</table>

### Video-to-Video Generation

This script can be used for various video-to-video tasks like video completion, video extension, or video transition, by providing an input video with at least 81 frames and specify condition settings. The generated video has 81 frames/21 latent frames in total.

**Example 1: Video Completion (Start-End Frames)**
Give the start frame and 4 end frames (encoded to one single latent frame) as conditions. 

```shell
CUDA_VISIBLE_DEVICES=0 python examples/pusavideo/wan_14b_v2v_pusa.py \
  --video_path "./demos/input_video.mp4" \
  --prompt "piggy bank surfing a tube in teahupo'o wave dusk light cinematic shot shot in 35mm film" \
  --cond_position "0,20" \
  --noise_multipliers "0.3,0.3" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt" \
  --lora_alpha 1.4 \
  --num_inference_steps 30
```

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_20_noise_0p0_0p0.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.0, 0.0]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_20_noise_0p2_0p2.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.2, 0.2]</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_20_noise_0p3_0p3.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.3, 0.3]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_20_noise_0p4_0p4.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.4, 0.4]</sub>
    </td>
  </tr>
</table>

**Example 2: Video Extension**
Give 13 frames as condition (encoded to the first 4 latent frames). 

```shell
CUDA_VISIBLE_DEVICES=0 python examples/pusavideo/wan_14b_v2v_pusa.py \
  --video_path "./demos/input_video.mp4" \
  --prompt "piggy bank surfing a tube in teahupo'o wave dusk light cinematic shot shot in 35mm film" \
  --cond_position "0,1,2,3" \
  --noise_multipliers "0.0,0.3,0.4,0.5" \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt" \
  --lora_alpha 1.4 \
  --num_inference_steps 30
```

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_1_2_3_noise_0p2_0p4_0p6_0p8.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.2, 0.4, 0.6, 0.8]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_1_2_3_noise_0p3_0p4_0p5_0p6.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.3, 0.4, 0.5, 0.6]</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_1_2_3_noise_0p0_0p3_0p4_0p5.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.0, 0.3, 0.4, 0.5]</sub>
    </td>
    <td align="center">
      <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/v2v_input_video_cond_0_1_2_3_noise_0p0_0p0_0p5_0p7.gif?raw=true" width="500"/>
      <br>
      <sub>noise: [0.0, 0.0, 0.5, 0.7]</sub>
    </td>
  </tr>
</table>

### Text-to-Video Generation

This script generates a video from a text prompt.

```shell
CUDA_VISIBLE_DEVICES=0 python examples/pusavideo/wan_14b_text_to_video_pusa.py \
  --prompt "A person is enjoying a meal of spaghetti with a fork in a cozy, dimly lit Italian restaurant. The person has warm, friendly features and is dressed casually but stylishly in jeans and a colorful sweater. They are sitting at a small, round table, leaning slightly forward as they eat with enthusiasm. The spaghetti is piled high on their plate, with some strands hanging over the edge. The background shows soft lighting from nearby candles and a few other diners in the corner, creating a warm and inviting atmosphere. The scene captures a close-up view of the person’s face and hands as they take a bite of spaghetti, with subtle movements of their mouth and fork. The overall style is realistic with a touch of warmth and authenticity, reflecting the comfort of a genuine dining experience." \
  --lora_path "./model_zoo/PusaV1/pusa_v1.pt" \
  --lora_alpha 1.4 \
  --num_inference_steps 30
```

<div align="center">
  <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/main/PusaV1/assets/t2v_output.gif?raw=true" width="500" autoplay loop muted controls></img>
</div>

## Training
First, please note that **our method also works for Full Finetuning with extremely low cost (see [Pusa V0.5](https://github.com/Yaofang-Liu/Mochi-Full-Finetuner), only $100 full finetuing cost). The model gains mainly come from our method, not Lora. Lora is insignificant here.**. We use Lora only because, Wan2.1 full finetuing need too many GPUs (at least 32x80G GPUs, yet we only have 24), so that we choose Lora with very large rank to approximate full finetuing. Why 512 rank is just because its basically the largest rank that can train with 8x80G GPUs. Besides, we also did full finetune on the base model with 81 frames in 480p data or 65 frames in 720p data with less GPUs, our method also works. **We suggest more to try full finetuing with our method if you have the resources. We believe the performance could be further imporved!**


Our training pipeline is based on [DiffySynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/3edf3583b1f08944cee837b94d9f84d669c2729c/examples/wanvideo), which supports both full finetuing and lora finetuning. We use LoRA training on a custom dataset to get Pusa V1.0 model. The training process consists of two stages: data preparation and training. 

### Prepare Dataset
You can download our dataset on Huggingface or prepare our own dataset following [DiffySynth-Studio](https://github.com/modelscope/DiffSynth-Studio/tree/3edf3583b1f08944cee837b94d9f84d669c2729c/examples/wanvideo).

Download `PusaV1_training` dataset to here `./dataset/`.
```shell
huggingface-cli download RaphaelLiu/PusaV1_training --repo-type dataset --local-dir ./dataset/
```

### Training
After prepraring the dataset, you can start training. We provide a sample script `train.sh` for multi-GPU training on a single node using `deepspeed` and Lora. 

You can find the content in `examples/pusavideo/train.sh` and modify the paths and parameters as needed. Finally, run the script from the `PusaV1` directory:
```shell
bash ./examples/pusavideo/train.sh
```
The trained LoRA model will be saved in the `lightning_logs` directory inside your specified `--output_path`.



