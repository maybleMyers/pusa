from PIL import Image
import torch
import os
import sys
import argparse
import datetime
import cv2
from diffsynth import ModelManagerWan22, Wan22VideoPusaV2VPipeline, save_video

# Add H1111 folder to path for block swapping utilities
sys.path.append(os.path.join(os.path.dirname(__file__), 'H1111'))

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

def enable_block_swapping_for_model(model, blocks_to_swap, device):
    """
    Enable block swapping for a model if it has the necessary methods.
    This is a simplified version that adds block swapping capability to existing models.
    """
    if blocks_to_swap <= 0:
        return

    # Check if model has blocks attribute (DiT models should have this)
    if not hasattr(model, 'blocks'):
        print(f"Warning: Model {type(model).__name__} does not have blocks attribute, skipping block swap")
        return

    print(f"Enabling block swap for {blocks_to_swap} blocks in {type(model).__name__}")

    # We'll implement a simple block swapping mechanism
    # Store references to blocks that will be swapped
    model.swap_blocks = []
    model.swap_device = device
    model.cpu_device = torch.device('cpu')

    # Select last N blocks to swap (typically the deepest blocks)
    total_blocks = len(model.blocks)
    if blocks_to_swap > total_blocks:
        blocks_to_swap = total_blocks
        print(f"Warning: Requested {blocks_to_swap} blocks but model only has {total_blocks}")

    # Mark blocks for swapping and move them to CPU
    for i in range(total_blocks - blocks_to_swap, total_blocks):
        block = model.blocks[i]
        model.swap_blocks.append((i, block))
        block.to('cpu')
        print(f"  Block {i} moved to CPU for swapping")

    # Store original forward method
    original_forward = model.forward

    def forward_with_swap(self, *args, **kwargs):
        # Move swap blocks to GPU before forward
        for idx, block in self.swap_blocks:
            block.to(self.swap_device)

        # Run original forward
        result = original_forward(*args, **kwargs)

        # Move swap blocks back to CPU after forward
        for idx, block in self.swap_blocks:
            block.to(self.cpu_device)

        return result

    # Replace forward method
    import types
    model.forward = types.MethodType(forward_with_swap, model)

    print(f"Block swapping enabled for {blocks_to_swap} blocks")

def main():
    parser = argparse.ArgumentParser(description="Pusa V2V with Block Swapping: Video-to-Video Generation with memory optimization")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the conditioning video.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation.")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", help="Negative text prompt for video generation.")
    parser.add_argument("--cond_position", type=str, default=None, help="Comma-separated list of frame indices for conditioning.")
    parser.add_argument("--extend_from_end", type=int, default=None, help="Number of frames from the end of `--video_path` to use for conditioning the start of the new video. Mutually exclusive with `--cond_position`.")
    parser.add_argument("--noise_multipliers", type=str, required=True, help="Comma-separated noise multipliers for conditioning frames.")
    parser.add_argument("--high_model_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/high_noise_model", help="Directory of the high noise DiT model components.")
    parser.add_argument("--low_model_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B/low_noise_model", help="Directory of the low noise DiT model components.")
    parser.add_argument("--base_dir", type=str, default="model_zoo/PusaV1/Wan2.2-T2V-A14B", help="Directory of the T2V model components (T5, VAE).")
    parser.add_argument("--high_lora_path", type=str, default="", help="Path(s) to the LoRA checkpoint file(s) for high noise model. Multiple paths separated by comma. Optional.")
    parser.add_argument("--high_lora_alpha", type=str, default="1.4", help="Alpha value(s) for high noise LoRA. Multiple values separated by comma.")
    parser.add_argument("--low_lora_path", type=str, default="", help="Path(s) to the LoRA checkpoint file(s) for low noise model. Multiple paths separated by comma. Optional.")
    parser.add_argument("--low_lora_alpha", type=str, default="1.4", help="Alpha value(s) for low noise LoRA. Multiple values separated by comma.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--switch_DiT_boundary", type=float, default=0.875, help="Boundary to switch between DiT models.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the output video.")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--lightx2v", action="store_true", help="Use lightx2v for acceleration.")
    parser.add_argument("--concatenate", action="store_true", help="Automatically concatenate the original video with the generated video for a final extended output. Only works with `--extend_from_end`.")
    parser.add_argument("--width", type=int, default=832, help="Width of the output video. Default: 832")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video. Default: 480")
    parser.add_argument("--fps", type=int, default=24, help="fps to save video in")

    # Block swapping memory management arguments
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of blocks to swap to CPU for memory management (0 = disabled)")

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

    # Parse multiple LoRA paths and alphas (handle empty strings)
    if args.high_lora_path and args.high_lora_path.strip():
        high_lora_paths = [path.strip() for path in args.high_lora_path.split(',') if path.strip()]
        high_lora_alphas = [float(alpha.strip()) for alpha in args.high_lora_alpha.split(',') if alpha.strip()]

        # Validate matching counts
        if len(high_lora_paths) != len(high_lora_alphas):
            raise ValueError(f"Number of high LoRA paths ({len(high_lora_paths)}) must match number of high LoRA alphas ({len(high_lora_alphas)})")

        # Load multiple LoRAs for high noise model
        for lora_path, lora_alpha in zip(high_lora_paths, high_lora_alphas):
            print(f"Loading high noise LoRA: {lora_path} with alpha={lora_alpha}")
            model_manager.load_loras_wan22(lora_path, lora_alpha=lora_alpha, model_type="high")
    else:
        print("No high noise LoRAs specified, using base model")

    if args.low_lora_path and args.low_lora_path.strip():
        low_lora_paths = [path.strip() for path in args.low_lora_path.split(',') if path.strip()]
        low_lora_alphas = [float(alpha.strip()) for alpha in args.low_lora_alpha.split(',') if alpha.strip()]

        # Validate matching counts
        if len(low_lora_paths) != len(low_lora_alphas):
            raise ValueError(f"Number of low LoRA paths ({len(low_lora_paths)}) must match number of low LoRA alphas ({len(low_lora_alphas)})")

        # Load multiple LoRAs for low noise model
        for lora_path, lora_alpha in zip(low_lora_paths, low_lora_alphas):
            print(f"Loading low noise LoRA: {lora_path} with alpha={lora_alpha}")
            model_manager.load_loras_wan22(lora_path, lora_alpha=lora_alpha, model_type="low")
    else:
        print("No low noise LoRAs specified, using base model")

    # Create pipeline
    pipe = Wan22VideoPusaV2VPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)

    # Enable block swapping if requested
    if args.blocks_to_swap > 0:
        print(f"\nEnabling block swapping for {args.blocks_to_swap} blocks...")

        # The pipeline should have dit and dit2 attributes for the two models
        if hasattr(pipe, 'dit'):
            enable_block_swapping_for_model(pipe.dit, args.blocks_to_swap, device)
        else:
            print("Warning: Pipeline does not have 'dit' attribute")

        if hasattr(pipe, 'dit2'):
            enable_block_swapping_for_model(pipe.dit2, args.blocks_to_swap, device)
        else:
            print("Warning: Pipeline does not have 'dit2' attribute")

        print("Block swapping configuration complete")
    else:
        # Use standard VRAM management if no block swapping
        pipe.enable_vram_management(num_persistent_param_in_dit=6e9)

    print(f"Models loaded successfully")

    # --- Prepare Conditioning Inputs ---
    all_video_frames = process_video_frames(args.video_path, target_width=args.width, target_height=args.height)
    print(f"Loaded {len(all_video_frames)} frames from video")

    # Determine which frames to use for conditioning
    if args.extend_from_end is not None:
        # Use frames from the end of the video
        cond_frames = all_video_frames[-args.extend_from_end:]
        cond_position = list(range(args.extend_from_end))
        print(f"Using last {args.extend_from_end} frames for conditioning at positions {cond_position}")
    else:
        # Use specified frame indices
        cond_position = [int(x.strip()) for x in args.cond_position.split(",")]
        cond_frames = [all_video_frames[min(i, len(all_video_frames)-1)] for i in cond_position]
        print(f"Using frames at indices {cond_position} for conditioning")

    noise_multipliers = [float(x.strip()) for x in args.noise_multipliers.split(",")]
    if len(noise_multipliers) != len(cond_frames):
        raise ValueError(f"Number of noise multipliers ({len(noise_multipliers)}) must match number of conditioning frames ({len(cond_frames)})")

    # --- Generate Video ---
    print(f"Generating video with prompt: {args.prompt}")
    print(f"Using {args.num_inference_steps} inference steps with switch boundary at {args.switch_DiT_boundary}")

    start_time = datetime.datetime.now()

    if args.lightx2v:
        video = pipe(
            prompt=args.prompt,
            conditioning_video=cond_frames,
            conditioning_indices=cond_position,
            conditioning_noise_multipliers=noise_multipliers,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=49,
            num_inference_steps=4,
            cfg_scale=1.0,
            switch_DiT_boundary=args.switch_DiT_boundary,
            use_lightx2v=True,
        )
    else:
        video = pipe(
            prompt=args.prompt,
            conditioning_video=cond_frames,
            conditioning_indices=cond_position,
            conditioning_noise_multipliers=noise_multipliers,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=49,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            switch_DiT_boundary=args.switch_DiT_boundary,
        )

    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"Video generation completed in {elapsed_time:.2f} seconds")

    # --- Save Output ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Create descriptive filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cond_str = f"cond_{cond_position}"
    noise_str = f"noise_{noise_multipliers}"
    if args.lightx2v:
        filename = f"wan22_v2v_blockswap_{timestamp}_{cond_str}_{noise_str}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}_lightx2v.mp4"
    else:
        filename = f"wan22_v2v_blockswap_{timestamp}_{cond_str}_{noise_str}_cfg_{args.cfg_scale}_steps_{args.num_inference_steps}.mp4"

    if args.blocks_to_swap > 0:
        filename = filename.replace(".mp4", f"_swap{args.blocks_to_swap}.mp4")

    # Handle concatenation if requested
    if args.concatenate and args.extend_from_end is not None:
        # Concatenate original video (minus overlap) with generated video
        original_frames = all_video_frames[:-args.extend_from_end] if args.extend_from_end < len(all_video_frames) else []
        final_video = original_frames + video
        filename = filename.replace(".mp4", "_concatenated.mp4")
        print(f"Concatenated {len(original_frames)} original frames with {len(video)} generated frames")
    else:
        final_video = video

    output_path = os.path.join(args.output_dir, filename)
    save_video(final_video, output_path, fps=args.fps)
    print(f"Video saved to: {output_path}")

    # Clean up memory if block swapping was used
    if args.blocks_to_swap > 0:
        print("Cleaning up memory...")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()