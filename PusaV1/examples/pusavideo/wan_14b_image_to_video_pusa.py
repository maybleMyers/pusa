from PIL import Image
import torch
import os
import argparse
from diffsynth import ModelManager, WanVideoPusaPipeline, save_video, VideoData
import datetime
import torchvision

def main():
    parser = argparse.ArgumentParser(description="Pusa I2V: Image-to-Video Generation")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint file.")
    parser.add_argument("--lora_alpha", type=float, default=1.4, help="LoRA alpha value.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the output video.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
    
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        ["model_zoo/PusaV1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,
    )
    
    base_dir = "model_zoo/PusaV1/Wan2.1-T2V-14B"
    model_files = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.safetensors')])

    model_manager.load_models(
        [
            model_files,
            os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(base_dir, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )

    model_manager.load_lora(args.lora_path, lora_alpha=args.lora_alpha)

    pipe = WanVideoPusaPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)

    num_inference_steps = 10
    seed = 0
    torch.manual_seed(seed)
    
    image = Image.open(args.image_path).convert("RGB")
    image = image.resize((1280, 720), Image.LANCZOS)
    video = pipe(
        prompt=args.prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        input_image=image,
        num_inference_steps=num_inference_steps,
        height=720, width=1280, num_frames=81,
        seed=seed, tiled=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.basename(args.image_path).split('.')[0]
    video_path = os.path.join(args.output_dir, f"i2v_{output_filename}_{timestamp}.mp4")
    
    if isinstance(video, list):
        video_tensor = torch.stack([torch.from_numpy(np.array(frame)) for frame in video])
    else:
        video_tensor = video
    torchvision.io.write_video(video_path, video_tensor, fps=25, video_codec='h264', options={'crf': '10'})
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()