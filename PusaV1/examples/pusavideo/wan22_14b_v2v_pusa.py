from PIL import Image
import torch
import os
import sys
import argparse
from diffsynth import ModelManagerWan22, Wan22VideoPusaV2VPipeline, save_video
import datetime
import cv2

def process_video_frames(video_path, target_width=832, target_height=480):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
        # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate scaling and cropping parameters
    target_ratio = target_width / target_height
    original_ratio = width / height
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize maintaining aspect ratio
        if original_ratio > target_ratio:
            # Video is wider than target
            new_width = int(height * target_ratio)
            # Crop width from center
            start_x = (width - new_width) // 2
            frame = frame[:, start_x:start_x + new_width]
        else:
            # Video is taller than target
            new_height = int(width / target_ratio)
            # Crop height from center
            start_y = (height - new_height) // 2
            frame = frame[start_y:start_y + new_height]
        
        # Resize to target dimensions
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Pusa V2V: Video-to-Video Generation with dual DiT models")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the conditioning video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, default=None, help="Comma-separated list of frame indices for conditioning.")
    parser.add_argument("--extend_from_end", type=int, default=None, help="Number of frames from the end of `--video_path` to use for conditioning the start of the new video. Mutually exclusive with `--cond_position`.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames.")
    parser.add_argument("--high_model_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/high_noise_model", help="Directory of the high noise DiT model components.")
    parser.add_argument("--low_model_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/low_noise_model", help="Directory of the low noise DiT model components.")
    parser.add_argument("--base_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B", help="Directory of the T2V model components (T5, VAE).")
    parser.add_argument("--high_lora_path", type=str, required=True, help="Path(s) to the LoRA checkpoint file(s) for high noise model. Multiple paths separated by comma.")
    parser.add_argument("--high_lora_alpha", type=str, default="1.4", help="Alpha value(s) for high noise LoRA. Multiple values separated by comma.")
    parser.add_argument("--low_lora_path", type=str, required=True, help="Path(s) to the LoRA checkpoint file(s) for low noise model. Multiple paths separated by comma.")
    parser.add_argument("--low_lora_alpha", type=str, default="1.4", help="Alpha value(s) for low noise LoRA. Multiple values separated by comma.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="Boundary to switch between DiT models.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    parser.add_argument("--concatenate", action="store_true", help="Automatically concatenate the original video with the generated video for a final extended output. Only works with `--extend_from_end`.")
    parser.add_argument("--num_persistent_params", type=float, default=6e9, help="Number of persistent parameters in DiT for VRAM management. Use scientific notation (e.g., 6e9 for 6 billion).")
    parser.add_argument("--width", type=int, default=832, help="Width of the output video. Default: 832")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video. Default: 480")
    args = parser.parse_args()

    # --- Argument Validation and Input Preparation ---
    if args.extend_from_end is not None and args.cond_position is not None:
        raise ValueError("Cannot use both `--extend_from_end` and `--cond_position`. Please choose one.")
    if args.extend_from_end is None and args.cond_position is None:
        raise ValueError("Either `--extend_from_end` or `--cond_position` must be specified for conditioning.")
    if args.concatenate and args.extend_from_end is None:
        raise ValueError("`--concatenate` can only be used with `--extend_from_end` for video extension.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Models ---
    print("Loading models...")
    model_manager = ModelManagerWan22(device="cpu")

    high_model_files = sorted([os.path.join(args.high_model_dir, f) for f in os.listdir(args.high_model_dir) if f.endswith('.safetensors')])
    low_model_files = sorted([os.path.join(args.low_model_dir, f) for f in os.listdir(args.low_model_dir) if f.endswith('.safetensors')])
    
    model_manager.load_models(
        [
            high_model_files,
            low_model_files,
            os.path.join(args.base_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(args.base_dir, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,
    )
    
    if args.lightx2v:
        # Lighx2v for acceleration
        high_lora_path = "./model_zoo/PusaV1/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(high_lora_path, model_type="high")
        low_lora_path = "./model_zoo/PusaV1/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
        model_manager.load_loras_wan22_lightx2v(low_lora_path, model_type="low")

    # Parse multiple LoRA paths and alphas
    high_lora_paths = [path.strip() for path in args.high_lora_path.split(',')]
    high_lora_alphas = [float(alpha.strip()) for alpha in args.high_lora_alpha.split(',')]
    low_lora_paths = [path.strip() for path in args.low_lora_path.split(',')]
    low_lora_alphas = [float(alpha.strip()) for alpha in args.low_lora_alpha.split(',')]

    # Validate matching counts
    if len(high_lora_paths) != len(high_lora_alphas):
        raise ValueError(f"Number of high LoRA paths ({len(high_lora_paths)}) must match number of high LoRA alphas ({len(high_lora_alphas)})")
    if len(low_lora_paths) != len(low_lora_alphas):
        raise ValueError(f"Number of low LoRA paths ({len(low_lora_paths)}) must match number of low LoRA alphas ({len(low_lora_alphas)})")

    # Load multiple LoRAs for high noise model
    for lora_path, lora_alpha in zip(high_lora_paths, high_lora_alphas):
        print(f"Loading high noise LoRA: {lora_path} with alpha={lora_alpha}")
        model_manager.load_loras_wan22(lora_path, lora_alpha=lora_alpha, model_type="high")

    # Load multiple LoRAs for low noise model
    for lora_path, lora_alpha in zip(low_lora_paths, low_lora_alphas):
        print(f"Loading low noise LoRA: {lora_path} with alpha={lora_alpha}")
        model_manager.load_loras_wan22(lora_path, lora_alpha=lora_alpha, model_type="low")
    
    pipe = Wan22VideoPusaV2VPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.enable_vram_management(num_persistent_param_in_dit=int(args.num_persistent_params))
    print(f"Models loaded successfully")
    
    # --- Prepare Conditioning Inputs ---
    all_video_frames = process_video_frames(args.video_path, target_width=args.width, target_height=args.height)
    noise_mult_list = [float(x.strip()) for x in args.noise_multipliers.split(',')]

    if args.extend_from_end:
        print(f"Video extension mode: Using last {args.extend_from_end} frames for conditioning.")
        if args.extend_from_end > len(all_video_frames):
            raise ValueError(f"`--extend_from_end` ({args.extend_from_end}) is greater than the number of frames in the video ({len(all_video_frames)}).")
        if len(noise_mult_list) != args.extend_from_end:
            raise ValueError(f"Number of noise multipliers ({len(noise_mult_list)}) must match `--extend_from_end` ({args.extend_from_end}).")
        
        # Use the last N frames of the input video
        conditioning_video = all_video_frames[-args.extend_from_end:]
        # Condition the first N frames of the output video
        cond_pos_list = list(range(args.extend_from_end))
    else:
        # Standard conditioning mode
        print(f"Standard conditioning mode: Using full video with specified positions.")
        cond_pos_list = [int(x.strip()) for x in args.cond_position.split(',')]
        if len(noise_mult_list) != len(cond_pos_list):
            raise ValueError(f"Number of noise multipliers ({len(noise_mult_list)}) must match the number of conditioning positions ({len(cond_pos_list)}).")
        conditioning_video = all_video_frames

    # --- Run Pipeline ---
    print("Generating new video frames...")
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        conditioning_video=conditioning_video,
        conditioning_indices=cond_pos_list,
        conditioning_noise_multipliers=noise_mult_list,
        num_inference_steps=args.num_inference_steps,
        height=args.height, width=args.width, num_frames=81,
        seed=0, tiled=True,
        switch_DiT_boundary=args.switch_DiT_boundary,
        cfg_scale=args.cfg_scale
    )
    print("Video generation complete.")
    
    # --- Save Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = os.path.basename(args.video_path).split('.')[0]
    
    # Create a descriptive filename stem
    if args.extend_from_end:
        mode_str = f"extend_{args.extend_from_end}frames"
    else:
        mode_str = f"cond_{str(cond_pos_list)}"
    
    base_video_filename = f"wan22_v2v_{output_filename_base}_{timestamp}_{mode_str}_noise_{str(noise_mult_list)}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}"
    if args.lightx2v:
        base_video_filename += "_lightx2v"
    
    # Decide which frames to save based on the --concatenate flag
    if args.concatenate and args.extend_from_end:
        print("Concatenating original video with the generated video...")
        final_video_frames = all_video_frames + video
        final_filename_stem = base_video_filename.replace(f"extend_{args.extend_from_end}frames", f"extended_total_{len(final_video_frames)}frames")
        video_to_save = final_video_frames
        video_filename = os.path.join(args.output_dir, final_filename_stem + ".mp4")
    else:
        video_to_save = video
        video_filename = os.path.join(args.output_dir, base_video_filename + ".mp4")

    print(f"Saving video to {video_filename}")
    save_video(video_to_save, video_filename, fps=24, quality=5)
    print("Video saved successfully.")

if __name__ == "__main__":
    main()