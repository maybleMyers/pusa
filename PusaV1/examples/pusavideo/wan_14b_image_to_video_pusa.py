from PIL import Image
import torch
import os
import sys
import argparse
from diffsynth import ModelManager, PusaMultiFramesPipeline, save_video
import datetime

def main():
    parser = argparse.ArgumentParser(description="Pusa Multi-Frame to Video Generation")
    parser.add_argument("--image_paths", type=str, nargs='+', required=True, help="Paths to the conditioning image frames.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, required=True, help="Comma-separated list of frame indices for conditioning.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames.")
    parser.add_argument("--i2v_model_path", type=str, default="model_zoo/PusaV1/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", help="Path to the I2V CLIP model.")
    parser.add_argument("--t2v_model_dir", type=str, default="model_zoo/PusaV1/Wan2.1-T2V-14B", help="Directory of the T2V model components.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint file.")
    parser.add_argument("--lora_alpha", type=float, default=1.4, help="Alpha value for LoRA.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the output video.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("Loading models...")
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [args.i2v_model_path],
        torch_dtype=torch.float32,
    )
    
    base_dir = args.t2v_model_dir
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
    
    pipe = PusaMultiFramesPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
    print(f"Models loaded successfully")

    cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]
    
    images = [Image.open(p).convert("RGB").resize((1280, 720), Image.LANCZOS) for p in args.image_paths]

    if len(images) != len(cond_pos_list) or len(images) != len(noise_mult_list):
        raise ValueError("The number of --image_paths, --cond_position, and --noise_multipliers must be the same.")

    multi_frame_images = {
        cond_pos: (img, noise_mult) 
        for cond_pos, img, noise_mult in zip(cond_pos_list, images, noise_mult_list)
    }

    video = pipe(
        prompt=args.prompt,
        negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        multi_frame_images=multi_frame_images,
        num_inference_steps=30,
        height=720, width=1280, num_frames=81,
        seed=0, tiled=True
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(args.output_dir, f"multi_frame_output_{timestamp}.mp4")
    print(f"Saved to {video_filename}")
    save_video(video, video_filename, fps=25, quality=5)

if __name__ == "__main__":
    main()