import argparse
from datetime import datetime
import gc
import random
import os
import re
import time
import math
from typing import Tuple, Optional, List, Union, Any
from pathlib import Path # Added for glob_images in V2V

import torch
import accelerate
from accelerate import Accelerator
from safetensors.torch import load_file as st_load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2 # Added for V2V video loading/resizing
import numpy as np # Added for V2V video processing
import torchvision.transforms.functional as TF
from tqdm import tqdm

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file, load_safetensors
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from blissful_tuner.latent_preview import LatentPreviewer

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device
# Original load_video/load_images are still needed for Fun-Control / image loading
# Copied and adapted from official generate.py's load_ref_images
from PIL import Image, ImageOps
import json # For sharded model loading

from hv_generate_video import save_images_grid, save_videos_grid, synchronize_device, load_images as hv_load_images, load_video as hv_load_video

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )
    parser.add_argument("--phantom_ckpt", type=str, default=None, help="Path to the Phantom-Wan DiT checkpoint (for S2V tasks). Can be .pth or .safetensors.")


    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path (for non-S2V tasks)")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is bfloat16")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_length", type=int, default=None, help="video length/number of frames. Default depends on task.")
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier free guidance (non-S2V tasks). Default is 5.0.",
    )
    # V2V arguments
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference (standard Wan V2V)")
    parser.add_argument("--strength", type=float, default=0.75, help="Strength for video2video inference (0.0-1.0)")
    # I2V / S2V arguments
    parser.add_argument("--ref_images", type=str, default=None, help="Comma-separated paths to reference images (for I2V or S2V tasks). For I2V, use a single path.")
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference (used with --ref_images for I2V start image)")

    # Fun-Control arguments (NEW/MODIFIED)
    parser.add_argument(
        "--control_path", # Keep this argument name
        type=str,
        default=None,
        help="path to control video for inference with Fun-Control model. video file or directory with images",
    )
    parser.add_argument(
        "--control_start",
        type=float,
        default=0.0,
        help="Start point (0.0-1.0) in the timeline where control influence is full (after fade-in)",
    )
    parser.add_argument(
        "--control_end",
        type=float,
        default=1.0,
        help="End point (0.0-1.0) in the timeline where control influence starts to fade out",
    )
    parser.add_argument(
        "--control_falloff_percentage", # NEW name
        type=float,
        default=0.3,
        help="Falloff percentage (0.0-0.49) for smooth transitions at start/end of control influence region",
    )
    parser.add_argument(
        "--control_weight", # NEW name
        type=float,
        default=1.0,
        help="Overall weight/strength of control video influence for Fun-Control (0.0 to high values)",
    )
    parser.add_argument("--trim_tail_frames", type=int, default=0, help="trim tail N frames from the video before saving")
    parser.add_argument(
        "--cfg_skip_mode",
        type=str,
        default="none",
        choices=["early", "late", "middle", "early_late", "alternate", "none"],
        help="CFG skip mode. each mode skips different parts of the CFG. "
        " early: initial steps, late: later steps, middle: middle steps, early_late: both early and late, alternate: alternate, none: no skip (default)",
    )
    parser.add_argument(
        "--cfg_apply_ratio",
        type=float,
        default=None,
        help="The ratio of steps to apply CFG (0.0 to 1.0). Default is None (apply all steps).",
    )
    parser.add_argument(
        "--slg_layers", type=str, default=None, help="Skip block (layer) indices for SLG (Skip Layer Guidance), comma separated"
    )
    parser.add_argument(
        "--slg_scale",
        type=float,
        default=3.0,
        help="scale for SLG classifier free guidance. Default is 3.0. Ignored if slg_mode is None or uncond",
    )
    parser.add_argument("--slg_start", type=float, default=0.0, help="start ratio for inference steps for SLG. Default is 0.0.")
    parser.add_argument("--slg_end", type=float, default=0.3, help="end ratio for inference steps for SLG. Default is 0.3.")
    parser.add_argument(
        "--slg_mode",
        type=str,
        default=None,
        choices=["original", "uncond"],
        help="SLG mode. original: same as SD3, uncond: replace uncond pred with SLG pred",
    )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Shift factor for flow matching schedulers. Default depends on task.",
    )
    # S2V Specific Arguments
    parser.add_argument("--guide_scale_img", type=float, default=5.0, help="CFG scale for reference images (S2V).")
    parser.add_argument("--guide_scale_text", type=float, default=7.5, help="CFG scale for text (S2V).")


    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        help="Torch.compile settings",
    )
    parser.add_argument("--preview", type=int, default=None, metavar="N",
        help="Enable latent preview every N steps. Generates previews in 'previews' subdirectory.",
    )
    parser.add_argument("--preview_suffix", type=str, default=None,
        help="Unique suffix for preview files to avoid conflicts in concurrent runs.",
    )

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    # Add checks for mutually exclusive arguments
    if args.video_path is not None and args.ref_images is not None:
        raise ValueError("--video_path and --ref_images cannot be used together.")
    if args.video_path is not None and args.control_path is not None:
        raise ValueError("--video_path (standard V2V) and --control_path (Fun-Control) cannot be used together.")
    if args.ref_images is not None and "t2v" in args.task and not "s2v" in args.task: # Allow ref_images for s2v
         logger.warning("--ref_images is provided, but task is set to t2v (and not s2v). This is unusual for standard T2V.")
    if args.control_path is not None and not WAN_CONFIGS[args.task].is_fun_control:
        raise ValueError("--control_path is provided, but the selected task does not support Fun-Control.")
    if not (0.0 <= args.control_falloff_percentage <= 0.49):
        raise ValueError("--control_falloff_percentage must be between 0.0 and 0.49")
    if args.task == "i2v-14B-FC-1.1" and args.ref_images is None:
         logger.warning(f"Task '{args.task}' typically uses --ref_images (single image) as the reference image for ref_conv. Proceeding without it.")
    if "s2v" in args.task and not args.phantom_ckpt:
        raise ValueError("--phantom_ckpt is required for S2V tasks.")
    if "s2v" in args.task and not args.ref_images:
        raise ValueError("--ref_images are required for S2V tasks.")
    if "s2v" in args.task and args.control_path:
        raise ValueError("S2V tasks do not support --control_path (Fun-Control) currently.")
    if "s2v" in args.task and args.video_path:
        raise ValueError("S2V tasks do not support --video_path (standard V2V) currently.")


    return args

def create_funcontrol_conditioning_latent(
    args: argparse.Namespace,
    config,
    vae: WanVAE,
    device: torch.device,
    lat_f: int,
    lat_h: int,
    lat_w: int,
    pixel_height: int, # Actual pixel height for resizing
    pixel_width: int   # Actual pixel width for resizing
) -> Optional[torch.Tensor]:
    """
    Creates the conditioning latent tensor 'y' for FunControl models,
    replicating the logic from WanWeightedControlToVideo node.

    Args:
        args: Command line arguments.
        config: Model configuration.
        vae: Loaded VAE model instance.
        device: Target computation device.
        lat_f: Number of latent frames.
        lat_h: Latent height.
        lat_w: Latent width.
        pixel_height: Target pixel height for image/video processing.
        pixel_width: Target pixel width for image/video processing.

    Returns:
        torch.Tensor: The final conditioning latent 'y' [1, 32, lat_f, lat_h, lat_w],
                      or None if VAE is missing when required.
    """
    logger.info("Creating FunControl conditioning latent 'y'...")
    if vae is None:
         # Should not happen if called correctly, but check anyway
         logger.error("VAE is required to create FunControl conditioning latent but was not provided.")
         return None

    batch_size = 1 # Hardcoded for script execution
    total_latent_frames = lat_f
    vae_dtype = vae.dtype # Use VAE's dtype for encoding

    # Initialize the two parts of the concat latent
    # Control part (first 16 channels) - will be filled later
    control_latent_part = torch.zeros([batch_size, 16, total_latent_frames, lat_h, lat_w],
                                    device=device, dtype=vae_dtype).contiguous()
    # Image guidance part (last 16 channels)
    image_guidance_latent = torch.zeros([batch_size, 16, total_latent_frames, lat_h, lat_w],
                                      device=device, dtype=vae_dtype).contiguous()

    # --- Image Guidance Processing (Start/End Images) ---
    timeline_mask = torch.zeros([1, 1, total_latent_frames], device=device, dtype=torch.float32).contiguous() # yapf: disable
    has_start_image = args.ref_images is not None and not "s2v" in args.task # S2V uses ref_images differently
    has_end_image = args.end_image_path is not None

    # Process start image if provided
    if has_start_image:
        start_image_path = args.ref_images.split(',')[0] # Use first image if multiple provided for I2V start
        logger.info(f"Processing start image for FunControl/I2V: {start_image_path}")
        try:
            img = Image.open(start_image_path).convert("RGB")
            img_np = np.array(img)
            # Resize to target pixel dimensions
            interpolation = cv2.INTER_AREA if pixel_height < img_np.shape[0] else cv2.INTER_CUBIC
            img_resized_np = cv2.resize(img_np, (pixel_width, pixel_height), interpolation=interpolation)
            # Convert to tensor CFHW, range [-1, 1]
            img_tensor = TF.to_tensor(img_resized_np).sub_(0.5).div_(0.5).to(device)
            img_tensor = img_tensor.unsqueeze(1) # Add frame dim: C,F,H,W

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae_dtype):
                # vae.encode expects a list, returns a list. Take first element.
                # Result shape [C', F', H', W'] - needs batch dim for processing here
                start_latent = vae.encode([img_tensor])[0].unsqueeze(0).to(device).contiguous() # [1, 16, 1, lat_h, lat_w]

            # Calculate influence and falloff
            start_frames_influence = min(start_latent.shape[2], total_latent_frames) # Usually 1
            if start_frames_influence > 0:
                 # Use falloff_percentage for smooth transition *away* from start image
                 falloff_len_frames = max(1, int(total_latent_frames * args.control_falloff_percentage))
                 start_influence_mask = torch.ones([1, 1, total_latent_frames], device=device, dtype=torch.float32).contiguous()

                 # Apply falloff starting *after* the first frame
                 if total_latent_frames > 1 + falloff_len_frames:
                     # Falloff from frame 1 to 1+falloff_len_frames
                     t = torch.linspace(0, 1, falloff_len_frames, device=device)
                     falloff = 0.5 + 0.5 * torch.cos(t * math.pi) # 1 -> 0
                     start_influence_mask[0, 0, 1:1+falloff_len_frames] = falloff
                     # Set influence to 0 after falloff
                     start_influence_mask[0, 0, 1+falloff_len_frames:] = 0.0
                 elif total_latent_frames > 1:
                     # Shorter falloff if video is too short
                     t = torch.linspace(0, 1, total_latent_frames - 1, device=device)
                     falloff = 0.5 + 0.5 * torch.cos(t * math.pi) # 1 -> 0
                     start_influence_mask[0, 0, 1:] = falloff

                 # Place start latent in the image guidance part, weighted by mask
                 # Since start_latent is only frame 0, we just place it there.
                 # The mask influences how other elements (like end image) blend *in*.
                 image_guidance_latent[:, :, 0:1, :, :] = start_latent[:, :, 0:1, :, :] # Take first frame

                 # Update the main timeline mask
                 timeline_mask = torch.max(timeline_mask, start_influence_mask) # Start image dominates beginning
                 logger.info(f"Start image processed. Latent shape: {start_latent.shape}")
                 del start_latent # free memory

        except Exception as e:
            logger.error(f"Error processing start image: {e}")
            # Continue without start image guidance

    # Process end image if provided
    end_latent = None
    if has_end_image:
        logger.info(f"Processing end image: {args.end_image_path}")
        try:
            img = Image.open(args.end_image_path).convert("RGB")
            img_np = np.array(img)
            # Resize to target pixel dimensions
            interpolation = cv2.INTER_AREA if pixel_height < img_np.shape[0] else cv2.INTER_CUBIC
            img_resized_np = cv2.resize(img_np, (pixel_width, pixel_height), interpolation=interpolation)
            # Convert to tensor CFHW, range [-1, 1]
            img_tensor = TF.to_tensor(img_resized_np).sub_(0.5).div_(0.5).to(device)
            img_tensor = img_tensor.unsqueeze(1) # Add frame dim: C,F,H,W

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae_dtype):
                # vae.encode expects a list, returns a list. Take first element.
                # Result shape [C', F', H', W'] - needs batch dim for processing here
                end_latent = vae.encode([img_tensor])[0].unsqueeze(0).to(device).contiguous() # [1, 16, 1, lat_h, lat_w]

            # Calculate end image influence transition (S-curve / cubic)
            end_influence_mask = torch.zeros([1, 1, total_latent_frames], device=device, dtype=torch.float32).contiguous()
            falloff_len_frames = max(1, int(total_latent_frames * args.control_falloff_percentage))

            # Determine when the end image influence should start ramping up
            # More sophisticated start point based on control_end if control video exists
            if args.control_path and args.control_end < 1.0:
                 # Start fade-in just before control video fades out significantly
                 influence_start_frame = max(0, int(total_latent_frames * args.control_end) - falloff_len_frames // 2)
            else:
                 # Default: start influence around 60% mark if no control or control runs full length
                 influence_start_frame = max(0, int(total_latent_frames * 0.6))

            # Ensure start frame isn't too close to the beginning if start image exists
            if has_start_image:
                influence_start_frame = max(influence_start_frame, 1 + falloff_len_frames) # Ensure it starts after start_img falloff

            transition_length = total_latent_frames - influence_start_frame
            if transition_length > 0:
                 logger.info(f"End image influence transition: frames {influence_start_frame} to {total_latent_frames-1}")
                 curve_positions = torch.linspace(0, 1, transition_length, device=device)
                 for i, pos in enumerate(curve_positions):
                     idx = influence_start_frame + i
                     if idx < total_latent_frames:
                         # Cubic ease-in-out curve (smoother than cosine)
                         if pos < 0.5: influence = 4 * pos * pos * pos
                         else: p = pos - 1; influence = 1 + 4 * p * p * p
                         # Ensure full influence near the end
                         if idx >= total_latent_frames - 3: influence = 1.0
                         end_influence_mask[0, 0, idx] = influence

                 # Blending logic (similar to base_nodes)
                 blend_start_frame = influence_start_frame
                 blend_length = total_latent_frames - blend_start_frame
                 if blend_length > 0:
                     # Create reference end latent (just the single frame repeated conceptually)
                     # Blend existing content with end latent based on influence weight
                     for i in range(blend_length):
                         idx = blend_start_frame + i
                         if idx < total_latent_frames:
                             weight = end_influence_mask[0, 0, idx].item()
                             if weight > 0:
                                 # Blend: (1-w)*current + w*end_latent
                                 image_guidance_latent[:, :, idx] = (
                                     (1.0 - weight) * image_guidance_latent[:, :, idx] +
                                     weight * end_latent[:, :, 0] # Use the single frame end_latent
                                 )

                 # Ensure final frames are exactly the end image latent
                 last_frames_exact = min(3, total_latent_frames) # Ensure at least last 3 frames are end image
                 if last_frames_exact > 0:
                     end_offset = total_latent_frames - last_frames_exact
                     if end_offset >= 0:
                         image_guidance_latent[:, :, end_offset:] = end_latent[:, :, 0:1].repeat(1, 1, last_frames_exact, 1, 1)

                 # Update the main timeline mask
                 timeline_mask = torch.max(timeline_mask, end_influence_mask)
                 logger.info(f"End image processed. Latent shape: {end_latent.shape}")
                 del end_latent # free memory

        except Exception as e:
            logger.error(f"Error processing end image: {e}")
            # Continue without end image guidance

    # --- Control Video Processing ---
    control_video_latent = None
    if args.control_path:
        logger.info(f"Processing control video: {args.control_path}")
        try:
            # Load control video frames (use helper from hv_generate_video for consistency)
            # Use args.video_length for the number of frames
            if os.path.isfile(args.control_path):
                video_frames_np = hv_load_video(args.control_path, 0, args.video_length, bucket_reso=(pixel_width, pixel_height))
            elif os.path.isdir(args.control_path):
                video_frames_np = hv_load_images(args.control_path, args.video_length, bucket_reso=(pixel_width, pixel_height))
            else:
                 raise FileNotFoundError(f"Control path not found: {args.control_path}")

            if not video_frames_np:
                raise ValueError("No frames loaded from control path.")

            num_control_frames_loaded = len(video_frames_np)
            if num_control_frames_loaded < args.video_length:
                 logger.warning(f"Control video loaded {num_control_frames_loaded} frames, less than target {args.video_length}. Padding with last frame.")
                 # Pad with the last frame
                 last_frame = video_frames_np[-1]
                 padding = [last_frame] * (args.video_length - num_control_frames_loaded)
                 video_frames_np.extend(padding)

            # Stack and convert to tensor: F, H, W, C -> B, C, F, H, W, range [-1, 1]
            video_frames_np = np.stack(video_frames_np[:args.video_length], axis=0) # Ensure correct length
            control_tensor = torch.from_numpy(video_frames_np).permute(0, 3, 1, 2).float() / 127.5 - 1.0 # F,C,H,W
            control_tensor = control_tensor.permute(1, 0, 2, 3) # C,F,H,W
            control_tensor = control_tensor.unsqueeze(0).to(device) # B,C,F,H,W

            # Encode control video
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae_dtype):
                # vae.encode expects list of [C, F, H, W], returns list of [C', F', H', W']
                control_video_latent = vae.encode([control_tensor[0]])[0].unsqueeze(0).to(device).contiguous() # [1, 16, lat_f, lat_h, lat_w]

            # Calculate weighted control mask (replicating base_nodes logic)
            control_frames_latent = control_video_latent.shape[2] # Should match total_latent_frames
            control_mask = torch.zeros([1, 1, control_frames_latent], device=device, dtype=torch.float32).contiguous()

            start_frame_idx = max(0, min(control_frames_latent - 1, int(control_frames_latent * args.control_start)))
            end_frame_idx = max(start_frame_idx + 1, min(control_frames_latent, int(control_frames_latent * args.control_end)))
            falloff_len_frames = max(2, int(control_frames_latent * args.control_falloff_percentage))

            # Main active region
            if start_frame_idx < end_frame_idx:
                control_mask[:, :, start_frame_idx:end_frame_idx] = 1.0

            # Fall-on at the start
            if start_frame_idx > 0:
                fallon_start = max(0, start_frame_idx - falloff_len_frames)
                fallon_len = start_frame_idx - fallon_start
                if fallon_len > 0:
                    t = torch.linspace(0, 1, fallon_len, device=device)
                    smooth_t = 0.5 - 0.5 * torch.cos(t * math.pi) # 0 -> 1
                    control_mask[:, :, fallon_start:start_frame_idx] = smooth_t.reshape(1, 1, -1)

            # Fall-off at the end (interacting with end_image influence)
            if end_frame_idx < control_frames_latent:
                falloff_start = end_frame_idx
                falloff_end = min(control_frames_latent, falloff_start + falloff_len_frames)
                falloff_actual_len = falloff_end - falloff_start
                if falloff_actual_len > 0:
                    # Check for end image influence in this region
                    if has_end_image:
                        for i in range(falloff_start, falloff_end):
                            # Calculate original falloff (1 -> 0)
                            fade_pos = (i - falloff_start) / falloff_actual_len
                            original_falloff = 0.5 + 0.5 * math.cos(fade_pos * math.pi)
                            # Get end image influence (already calculated in timeline_mask)
                            end_influence_here = timeline_mask[0, 0, i].item()
                            # Adjust control falloff: decrease faster if end image is taking over
                            # Use a factor (e.g., 0.8) to control how much end image preempts control
                            adjusted_falloff = original_falloff * (1.0 - (end_influence_here * 0.8))
                            control_mask[0, 0, i] = adjusted_falloff
                        logger.info("Applied end-image interaction to control falloff.")
                    else:
                        # Standard falloff if no end image
                        t = torch.linspace(0, 1, falloff_actual_len, device=device)
                        smooth_t = 0.5 + 0.5 * torch.cos(t * math.pi) # 1 -> 0
                        control_mask[:, :, falloff_start:falloff_end] = smooth_t.reshape(1, 1, -1)

            # Apply final control weight
            control_mask = control_mask * args.control_weight

            # Expand mask and apply to control latent
            control_mask_expanded = control_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, 1, 1, F, 1, 1] ? -> needs [1, 1, F, 1, 1]
            control_mask_expanded = control_mask.unsqueeze(-1).unsqueeze(-1) # Shape: [1, 1, F, 1, 1]

            # Apply weighting to the control_video_latent
            weighted_control_latent = control_video_latent * control_mask_expanded # [1, 16, F, H, W]

            # Place into the first 16 channels of the final latent
            control_latent_part = weighted_control_latent

            # Log mask pattern
            mask_pattern = "".join(["#" if v > 0.8*args.control_weight else "+" if v > 0.4*args.control_weight else "." if v > 0.1*args.control_weight else " "
                                   for v in control_mask[0, 0, :].tolist()])
            logger.info(f"Control mask pattern (weight={args.control_weight:.2f}): |{mask_pattern}|")
            logger.info(f"Control video processed. Latent shape: {control_video_latent.shape}")
            del control_video_latent # free memory
            if 'control_tensor' in locals(): del control_tensor


        except Exception as e:
            logger.error(f"Error processing control video: {e}")
            # Continue without control video guidance (control_latent_part remains zeros)

    # --- Final Assembly ---
    # Concatenate the control part and the image guidance part
    final_y = torch.cat([control_latent_part, image_guidance_latent], dim=1) # Concat along channel dim: [1, 16+16, F, H, W]
    final_y = final_y.contiguous()

    logger.info(f"FunControl conditioning latent 'y' created. Final shape: {final_y.shape}")

    # Optional: Clean up intermediate tensors explicitly if memory is tight
    del control_latent_part, image_guidance_latent
    del timeline_mask, control_mask
    if 'img_tensor' in locals(): del img_tensor
    clean_memory_on_device(device) # Be cautious with frequent cache clearing

    return final_y

def get_task_defaults(task: str, size: Optional[Tuple[int, int]] = None) -> Tuple[int, float, int, bool]:
    """Return default values for each task

    Args:
        task: task name (t2v, t2i, i2v etc.)
        size: size of the video (width, height)

    Returns:
        Tuple[int, float, int, bool]: (infer_steps, flow_shift, video_length, needs_clip)
    """
    width, height = size if size else (0, 0)

    if "t2i" in task:
        return 50, 5.0, 1, False
    elif "i2v" in task: # Standard I2V
        flow_shift = 3.0 if (width == 832 and height == 480) or (width == 480 and height == 832) else 5.0 # yapf: disable
        return 40, flow_shift, 81, True
    elif "s2v" in task: # S2V defaults
        # S2V defaults from official generate.py: steps=50, shift=5.0 (or 3.0 for 832*480/480*832), frame_num=81
        # Note: Official generate.py sets sample_shift = 3.0 for s2v on "832*480", "480*832",
        # but the example line uses 832*480 with default shift (which becomes 5.0).
        # We will match the general default from other tasks unless specified by args.
        # The `frame_num` (video_length) is set to 81 by default in its own block.
        flow_shift = 3.0 if (width == 832 and height == 480) or (width == 480 and height == 832) else 5.0
        return 50, flow_shift, 81, False # S2V does not use CLIP for image features
    else:  # t2v or default
        return 50, 5.0, 81, False


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and set default values for optional arguments

    Args:
        args: command line arguments

    Returns:
        argparse.Namespace: updated arguments
    """
    # Get default values for the task
    infer_steps, flow_shift, video_length, _ = get_task_defaults(args.task, tuple(args.video_size))

    # Apply default values to unset arguments
    if "s2v" in args.task: # S2V has its own defaults from official generate.py
        if args.infer_steps is None: args.infer_steps = 50
        if args.flow_shift is None: args.flow_shift = 3.0 if (args.video_size[1] == 832 and args.video_size[0] == 480) or \
                                                              (args.video_size[1] == 480 and args.video_size[0] == 832) else 5.0
        if args.video_length is None: args.video_length = 81 # Default from generate.py (frame_num)
        # S2V specific guidance scales already have defaults in argparse
    else:
        if args.infer_steps is None:
            args.infer_steps = infer_steps
        if args.flow_shift is None:
            args.flow_shift = flow_shift

    # For V2V, video_length might be determined by the input video later if not set
    if args.video_length is None and args.video_path is None:
        if not ("s2v" in args.task and args.video_length is not None): # Don't override S2V default if already set
            args.video_length = video_length
    elif args.video_length is None and args.video_path is not None:
        # Delay setting default if V2V and length not specified
        pass
    elif args.video_length is not None:
        # Use specified length
        pass


    # Force video_length to 1 for t2i tasks
    if "t2i" in args.task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"

    # parse slg_layers
    if args.slg_layers is not None:
        args.slg_layers = list(map(int, args.slg_layers.split(",")))

    return args


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, Optional[int]]:
    """Validate video size and potentially length (if not V2V auto-detect)

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, Optional[int]]: (height, width, video_length)
    """
    height = args.video_size[0]
    width = args.video_size[1]
    size = f"{width}*{height}"

    # Only check supported sizes if not doing V2V (V2V might use custom sizes from input),
    # Or if it's FunControl, which might have different size constraints
    # S2V also has its own supported sizes in SUPPORTED_SIZES
    # Standard T2V/I2V checks against SUPPORTED_SIZES
    is_standard_task_for_size_check = not args.video_path and \
                                      not WAN_CONFIGS[args.task].is_fun_control
    if is_standard_task_for_size_check or "s2v" in args.task: # Include S2V in size check
        if args.task in SUPPORTED_SIZES and size not in SUPPORTED_SIZES[args.task]:
            logger.warning(f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}.")


    video_length = args.video_length # Might be None if V2V auto-detect

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    return height, width, video_length


def calculate_dimensions(video_size: Tuple[int, int], video_length: int, config) -> Tuple[Tuple[int, int, int, int], int]:
    """calculate dimensions for the generation

    Args:
        video_size: video frame size (height, width)
        video_length: number of frames in the video
        config: model configuration

    Returns:
        Tuple[Tuple[int, int, int, int], int]:
            ((channels, frames, height, width), seq_len)
    """
    height, width = video_size
    frames = video_length

    # calculate latent space dimensions
    lat_f = (frames - 1) // config.vae_stride[0] + 1
    lat_h = height // config.vae_stride[1]
    lat_w = width // config.vae_stride[2]

    # calculate sequence length
    seq_len = math.ceil((lat_h * lat_w) / (config.patch_size[1] * config.patch_size[2]) * lat_f)
    
    # For S2V, VAE output channels are used for DiT input channels
    dit_input_channels = getattr(config, 'in_channels', config.vae_z_dim if hasattr(config, 'vae_z_dim') else 16)


    return ((dit_input_channels, lat_f, lat_h, lat_w), seq_len)


# Modified function (replace the original)
def load_vae(args: argparse.Namespace, config, device: torch.device, dtype: torch.dtype) -> WanVAE:
    """load VAE model with robust path handling

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dtype: data type for the model

    Returns:
        WanVAE: loaded VAE model
    """
    vae_override_path = args.vae
    vae_filename = config.vae_checkpoint # Get expected filename, e.g., "Wan2.1_VAE.pth"
    # Assume models are in 'wan' dir relative to script if not otherwise specified
    vae_base_dir = "wan"

    final_vae_path = None

    # 1. Check if args.vae is a valid *existing file path*
    if vae_override_path and isinstance(vae_override_path, str) and \
       (vae_override_path.endswith(".pth") or vae_override_path.endswith(".safetensors")) and \
       os.path.isfile(vae_override_path):
        final_vae_path = vae_override_path
        logger.info(f"Using VAE override path from --vae: {final_vae_path}")

    # 2. If override is invalid or not provided, construct default path
    if final_vae_path is None:
        constructed_path = os.path.join(vae_base_dir, vae_filename)
        if os.path.isfile(constructed_path):
            final_vae_path = constructed_path
            logger.info(f"Constructed default VAE path: {final_vae_path}")
            if vae_override_path:
                 logger.warning(f"Ignoring potentially invalid --vae argument: {vae_override_path}")
        else:
             # 3. Fallback using ckpt_dir if provided and default construction failed
             if args.ckpt_dir:
                 fallback_path = os.path.join(args.ckpt_dir, vae_filename)
                 if os.path.isfile(fallback_path):
                     final_vae_path = fallback_path
                     logger.info(f"Using VAE path from --ckpt_dir fallback: {final_vae_path}")
                 else:
                     # If all attempts fail, raise error
                     raise FileNotFoundError(f"Cannot find VAE. Checked override '{vae_override_path}', constructed '{constructed_path}', and fallback '{fallback_path}'")
             else:
                 raise FileNotFoundError(f"Cannot find VAE. Checked override '{vae_override_path}' and constructed '{constructed_path}'. No --ckpt_dir provided for fallback.")

    # At this point, final_vae_path should be valid
    logger.info(f"Loading VAE model from final path: {final_vae_path}")
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    vae = WanVAE(vae_path=final_vae_path, device=device, dtype=dtype, cache_device=cache_device)
    return vae


def load_text_encoder(args: argparse.Namespace, config, device: torch.device) -> T5EncoderModel:
    """load text encoder (T5) model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        T5EncoderModel: loaded text encoder model
    """
    checkpoint_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_checkpoint)
    tokenizer_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_tokenizer)

    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        weight_path=args.t5,
        fp8=args.fp8_t5,
    )

    return text_encoder


def load_clip_model(args: argparse.Namespace, config, device: torch.device) -> CLIPModel:
    """load CLIP model (for I2V only)

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        CLIPModel: loaded CLIP model
    """
    checkpoint_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.clip_checkpoint)
    tokenizer_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.clip_tokenizer)

    clip = CLIPModel(
        dtype=config.clip_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        weight_path=args.clip,
    )

    return clip


def load_dit_model(
    args: argparse.Namespace,
    config,
    device: torch.device,
    dit_dtype: torch.dtype,
    dit_weight_dtype: Optional[torch.dtype] = None,
    is_i2v: bool = False, # is_i2v might influence model loading specifics in some versions
) -> WanModel:
    """load DiT model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is
        is_i2v: I2V mode (might affect some model config details)

    Returns:
        WanModel: loaded DiT model
    """
    is_s2v_task = "s2v" in args.task
    if is_s2v_task and args.phantom_ckpt:
        loading_device = "cpu" # Phantom model loaded to CPU first
    else:
        loading_device = "cpu" # Standard Wan loading to CPU first

    if args.blocks_to_swap == 0 and args.lora_weight is None and not args.fp8_scaled:
        loading_device = device

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled or args.lora_weight is not None:
        loading_weight_dtype = dit_dtype  # load as-is

    if is_s2v_task and args.phantom_ckpt:
        logger.info(f"Loading Phantom DiT for S2V task from: {args.phantom_ckpt}")
        # Ensure S2V config has vae_z_dim, or default for in_channels
        vae_z_dim_s2v = getattr(config, 'vae_z_dim', 16) # Default to 16 if not in config
        
        # Get patch_size from config or default
        s2v_patch_size = getattr(config, 'patch_size', (1,2,2))
        
        # Determine model_type for S2V
        # Since S2V concatenates VAE latents and uses text CFG, 't2v' model_type is appropriate
        # as it will use WanT2VCrossAttention and not expect CLIP image features directly.
        s2v_model_type = "t2v"

        model = WanModel(
            model_type=s2v_model_type, # Explicitly set model_type
            dim=config.dim, 
            ffn_dim=config.ffn_dim, 
            freq_dim=config.freq_dim,
            num_heads=config.num_heads, 
            num_layers=config.num_layers,
            window_size=config.window_size, 
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm, 
            eps=config.eps,
            in_dim=vae_z_dim_s2v, # Changed from in_channels
            out_dim=vae_z_dim_s2v, # Changed from out_channels
            patch_size=s2v_patch_size, # Changed from patch_size_t and provide full tuple
            text_dim=getattr(config, 'text_dim', 4096),
            attn_mode=args.attn_mode,
            # add_ref_conv and in_dim_ref_conv will be handled by load_wan_model if this path merged
            # For direct S2V init, if phantom model could have ref_conv, this needs to be detected from ckpt
            # Assuming current phantom S2V doesn't use ref_conv as per subject2video.py init
            add_ref_conv=False, 
        )
        if args.phantom_ckpt.endswith('.pth'):
            state = torch.load(args.phantom_ckpt, map_location=loading_device)
            # Remove "model.diffusion_model." prefix if present, common in some checkpoints
            # Also, subject2video.py loads state directly without this prefix.
            # And WanModel in model.py doesn't have "model.diffusion_model." in its keys.
            # So, if state from .pth has it, it should be removed.
            # However, the provided subject2video.py loads it directly, so assume it's not needed for pth.
            # If loading a general .pth, this might be needed:
            # new_state_dict = {}
            # for k, v in state.items():
            #     if k.startswith("model.diffusion_model."):
            #         new_state_dict[k[len("model.diffusion_model."):]] = v
            #     else:
            #         new_state_dict[k] = v
            # state = new_state_dict
            model.load_state_dict(state, strict=False)
        elif args.phantom_ckpt.endswith('.safetensors'):
            # Check if it's a sharded model index or a single file
            if os.path.exists(args.phantom_ckpt + ".index.json"): # Sharded model
                 logger.info(f"Loading sharded Phantom-Wan DiT from directory of: {args.phantom_ckpt}")
                 from safetensors.torch import load_file
                 import json
                 def load_custom_sharded_weights(model_dir, base_name, device="cpu"):
                     index_path = os.path.join(model_dir, f"{base_name}.safetensors.index.json")
                     if not os.path.exists(index_path): # Try to find index file if base_name is full path
                        model_dir_abs, base_name_file = os.path.split(base_name)
                        index_path_alt = os.path.join(model_dir_abs, f"{os.path.splitext(base_name_file)[0]}.index.json")
                        if os.path.exists(index_path_alt):
                            index_path = index_path_alt
                            logger.info(f"Found sharded index at: {index_path}")
                        else:
                             raise FileNotFoundError(f"Sharded index file not found at {index_path} or {index_path_alt}")

                     with open(index_path, "r") as f:
                         index_data = json.load(f)
                     weight_map = index_data["weight_map"]
                     shard_files = set(weight_map.values())
                     state_dict = {}
                     actual_model_dir = os.path.dirname(index_path) # Directory where shards are located
                     for shard_file in shard_files:
                         shard_path = os.path.join(actual_model_dir, shard_file)
                         shard_state = load_file(shard_path, device=loading_device) 
                         state_dict.update(shard_state)
                     return state_dict
                 
                 ckpt_dir = os.path.dirname(args.phantom_ckpt)
                 ckpt_basename = os.path.basename(args.phantom_ckpt)
                 # Pass the full path of the .safetensors file as base_name, load_custom_sharded_weights will parse it
                 state = load_custom_sharded_weights(ckpt_dir, args.phantom_ckpt, device=loading_device)

            else: # Single safetensors file
                 state = st_load_file(args.phantom_ckpt, device=loading_device)

            # Remove "model.diffusion_model." prefix (same logic as .pth, but less common for safetensors)
            # new_state_dict = {}
            # for k, v in state.items():
            #     if k.startswith("model.diffusion_model."):
            #         new_state_dict[k[len("model.diffusion_model."):]] = v
            #     else:
            #         new_state_dict[k] = v
            # state = new_state_dict
            model.load_state_dict(state, strict=False) # Use strict=False initially
        else:
            raise ValueError(f"Unsupported Phantom-Wan checkpoint format: {args.phantom_ckpt}. Must be .pth or .safetensors.")
        logger.info("Phantom DiT model loaded.")
    else:
        # Standard Wan DiT loading
        # do not fp8 optimize because we will merge LoRA weights
        # The 'is_i2v' flag might be used internally by load_wan_model if needed by specific Wan versions
        model = load_wan_model(config, device, args.dit, args.attn_mode, False, loading_device, loading_weight_dtype, False)


    return model


def merge_lora_weights(model: WanModel, args: argparse.Namespace, device: torch.device) -> None:
    """merge LoRA weights to the model

    Args:
        model: DiT model
        args: command line arguments
        device: device to use
    """
    if args.lora_weight is None or len(args.lora_weight) == 0:
        return

    for i, lora_weight in enumerate(args.lora_weight):
        if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
            lora_multiplier = args.lora_multiplier[i]
        else:
            lora_multiplier = 1.0

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = st_load_file(lora_weight)

        # apply include/exclude patterns
        original_key_count = len(weights_sd.keys())
        if args.include_patterns is not None and len(args.include_patterns) > i:
            include_pattern = args.include_patterns[i]
            regex_include = re.compile(include_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
            logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")
        if args.exclude_patterns is not None and len(args.exclude_patterns) > i:
            original_key_count_ex = len(weights_sd.keys())
            exclude_pattern = args.exclude_patterns[i]
            regex_exclude = re.compile(exclude_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
            logger.info(
                f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
            )
        if len(weights_sd) != original_key_count:
            remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
            remaining_keys.sort()
            logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
            if len(weights_sd) == 0:
                logger.warning(f"No keys left after filtering.")

        if args.lycoris:
            lycoris_net, _ = create_network_from_weights(
                multiplier=lora_multiplier,
                file=None,
                weights_sd=weights_sd,
                unet=model,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            lycoris_net.merge_to(None, model, weights_sd, dtype=None, device=device)
        else:
            network = lora_wan.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=model, for_inference=True)
            network.merge_to(None, model, weights_sd, device=device, non_blocking=True)

        synchronize_device(device)
        logger.info("LoRA weights loaded")

    # save model here before casting to dit_weight_dtype
    if args.save_merged_model:
        logger.info(f"Saving merged model to {args.save_merged_model}")
        mem_eff_save_file(model.state_dict(), args.save_merged_model)  # save_file needs a lot of memory
        logger.info("Merged model saved")


def optimize_model(
    model: WanModel, args: argparse.Namespace, device: torch.device, dit_dtype: torch.dtype, dit_weight_dtype: torch.dtype
) -> None:
    """optimize the model (FP8 conversion, device move etc.)

    Args:
        model: dit model
        args: command line arguments
        device: device to use
        dit_dtype: dtype for the model
        dit_weight_dtype: dtype for the model weights
    """
    if args.fp8_scaled:
        # load state dict as-is and optimize to fp8
        state_dict = model.state_dict()

        # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
        move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
        state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded FP8 optimized weights: {info}")

        if args.blocks_to_swap == 0:
            model.to(device)  # make sure all parameters are on the right device (e.g. RoPE etc.)
    else:
        # simple cast to dit_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    if args.compile:
        compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
        logger.info(
            f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
        )
        torch._dynamo.config.cache_size_limit = 32
        for i in range(len(model.blocks)):
            model.blocks[i] = torch.compile(
                model.blocks[i],
                backend=compile_backend,
                mode=compile_mode,
                dynamic=compile_dynamic.lower() in "true",
                fullgraph=compile_fullgraph.lower() in "true",
            )

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)


def prepare_t2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: Optional[WanVAE] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for T2V (including Fun-Control variation)

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, required only for Fun-Control

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, (arg_c, arg_null))
    """
    # Prepare inputs for T2V
    # calculate dimensions and sequence length
    height, width = args.video_size
    frames = args.video_length # Should be set by now
    (ch, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, args.video_length, config)
    target_shape = (ch, lat_f, lat_h, lat_w) # Should be (16, lat_f, lat_h, lat_w) for base latent

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # set seed
    seed = args.seed # Seed should be set in generate()
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # ComfyUI compatible noise
        seed_g = torch.manual_seed(seed)

    # load text encoder
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)

    # encode prompt
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)

    # free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)

    # Initialize 'y' (conditioning latent) to None
    y = None

    # Handle Fun-Control T2V case
    if config.is_fun_control:
        logger.info("Preparing inputs for Fun-Control T2V.")
        if vae is None:
            raise ValueError("VAE is required for Fun-Control T2V input preparation.")

        # Calculate pixel dimensions needed for encoding helper
        pixel_height = lat_h * config.vae_stride[1]
        pixel_width = lat_w * config.vae_stride[2]

        # Create the conditioning latent 'y'
        # This function handles control video encoding (if path provided)
        # and creates the [1, 32, F, H, W] tensor.
        # If no control path, it creates the control part as zeros.
        # Since this is T2V, image_path and end_image_path are None in args,
        # so the image guidance part will also be zeros.
        vae.to_device(device) # Ensure VAE is on device
        y = create_funcontrol_conditioning_latent(
            args, config, vae, device, lat_f, lat_h, lat_w, pixel_height, pixel_width
        )
        # Move VAE back after use
        vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
        clean_memory_on_device(device)

        if y is None:
            raise RuntimeError("Failed to create FunControl conditioning latent 'y'.")

    # generate noise (base latent noise, shape [16, F, H, W])
    noise = torch.randn(target_shape, dtype=torch.float32, generator=seed_g, device=device if not args.cpu_noise else "cpu")
    noise = noise.to(device)

    # prepare model input arguments
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    # Add 'y' ONLY if it was created (i.e., for Fun-Control)
    if y is not None:
        arg_c["y"] = [y] # Model expects y as a list
        arg_null["y"] = [y]
        logger.info(f"Added FunControl conditioning 'y' (shape: {y.shape}) to model inputs.")
    elif config.is_fun_control:
        # This case should technically be handled by y being zeros, but double-check
         logger.warning("FunControl task but 'y' tensor was not generated. Model might error.")
         # Create a zero tensor as fallback if y generation failed somehow?
         # y = torch.zeros([1, 32, lat_f, lat_h, lat_w], device=device, dtype=noise.dtype)
         # arg_c["y"] = [y]
         # arg_null["y"] = [y]


    return noise, context, context_null, (arg_c, arg_null)

# ========================================================================= #
# START OF MODIFIED FUNCTION prepare_i2v_inputs
# ========================================================================= #
def prepare_i2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: WanVAE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for I2V (including Fun-Control I2V variation)

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, used for image encoding

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, y, (arg_c, arg_null))
            'y' is the conditioning latent ([1, 32, F, H, W] for FunControl,
             [1, C+4, F, H, W] for standard I2V with mask).
    """
    if vae is None:
        raise ValueError("VAE must be provided for I2V input preparation.")

    # --- Prepare Conditioning Latent 'y' ---
    # This check MUST come first to decide the entire logic path
    if config.is_fun_control:
        # --- FunControl I2V Path ---
        logger.info("Preparing inputs for Fun-Control I2V.")

        # Calculate dimensions (FunControl might use different aspect logic)
        height, width = args.video_size
        frames = args.video_length # Should be set by now
        (_, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, args.video_length, config)
        pixel_height = lat_h * config.vae_stride[1]
        pixel_width = lat_w * config.vae_stride[2]
        noise_channels = 16 # FunControl DiT denoises 16 channels

        logger.info(f"FunControl I2V target pixel resolution: {pixel_height}x{pixel_width}, latent shape: ({lat_f}, {lat_h}, {lat_w}), seq_len: {seq_len}")

        # set seed
        seed = args.seed
        if not args.cpu_noise:
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(seed)
        else:
            seed_g = torch.manual_seed(seed)

        # generate noise (for the part being denoised by the DiT)
        noise = torch.randn(
            noise_channels, lat_f, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g,
            device=device if not args.cpu_noise else "cpu",
        )
        noise = noise.to(device)

        # configure negative prompt
        n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

        # load text encoder & encode prompts
        text_encoder = load_text_encoder(args, config, device)
        text_encoder.model.to(device)
        with torch.no_grad():
            if args.fp8_t5:
                with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                    context = text_encoder([args.prompt], device)
                    context_null = text_encoder([n_prompt], device)
            else:
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        del text_encoder
        clean_memory_on_device(device)

        # load CLIP model & encode image
        clip = load_clip_model(args, config, device)
        clip.model.to(device)
        if not args.ref_images: # Changed from image_path
             raise ValueError("--ref_images (single image) is required for FunControl I2V mode.")
        start_image_path_fc = args.ref_images.split(',')[0] # Use first image
        img_clip = Image.open(start_image_path_fc).convert("RGB")
        img_tensor_clip = TF.to_tensor(img_clip).sub_(0.5).div_(0.5).to(device) # yapf: disable # CHW, [-1, 1]
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
            clip_context = clip.visual([img_tensor_clip.unsqueeze(1)]) # Add Frame dim
        del clip, img_clip, img_tensor_clip
        clean_memory_on_device(device)

        fun_ref_latent = None
        # Check if the task requires ref_conv and if a reference image is provided via --ref_images
        if args.task == "i2v-14B-FC-1.1" and args.ref_images is not None:
            ref_image_path_for_conv = args.ref_images.split(',')[0]
            logger.info(f"Task {args.task} requires ref_conv. Encoding reference image from --ref_images: {ref_image_path_for_conv}")
            try:
                ref_img_pil = Image.open(ref_image_path_for_conv).convert("RGB")
                ref_img_np = np.array(ref_img_pil)
                # Resize ref image to target pixel dimensions
                interpolation = cv2.INTER_AREA if pixel_height < ref_img_np.shape[0] else cv2.INTER_CUBIC
                ref_img_resized_np = cv2.resize(ref_img_np, (pixel_width, pixel_height), interpolation=interpolation)
                # Convert to tensor CFHW, range [-1, 1]
                ref_img_tensor = TF.to_tensor(ref_img_resized_np).sub_(0.5).div_(0.5).to(device)
                ref_img_tensor = ref_img_tensor.unsqueeze(1) # Add frame dim: C,F,H,W

                vae.to_device(device) # Ensure VAE is on device for encoding
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype):
                    # Encode the single reference frame
                    # vae.encode returns list, take first element. Result shape [C', 1, H', W']
                    fun_ref_latent = vae.encode([ref_img_tensor])[0]
                    # Squeeze the frame dimension for Conv2d in the model: [C', H', W']
                    fun_ref_latent = fun_ref_latent.squeeze(1)
                logger.info(f"Encoded fun_ref latent. Shape: {fun_ref_latent.shape}")
                # Keep VAE on device for main conditioning latent creation below

            except Exception as e:
                logger.error(f"Error processing reference image for fun_ref: {e}")
                fun_ref_latent = None # Continue without ref if encoding fails

            # **IMPORTANT**: Since --ref_images is now used for fun_ref,
            # temporarily set it to None *before* calling create_funcontrol_conditioning_latent
            # so it doesn't get processed *again* as a start image inside that function if logic relies on it.
            original_ref_images_arg = args.ref_images # Save original
            args.ref_images = None # Temporarily clear for create_funcontrol_conditioning_latent

        # Use the FunControl helper function to create the 32-channel 'y'
        # Crucially, create_funcontrol_conditioning_latent itself uses args.ref_images (if set)
        # for its *own* start image logic, which is separate from the fun_ref above.
        vae.to_device(device) # Ensure VAE is on compute device
        y = create_funcontrol_conditioning_latent(
            args, config, vae, device, lat_f, lat_h, lat_w, pixel_height, pixel_width
        )
        if args.task == "i2v-14B-FC-1.1": # Restore original args.ref_images if it was modified
             args.ref_images = original_ref_images_arg
        if y is None:
            raise RuntimeError("Failed to create FunControl conditioning latent 'y'.")
        vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu") # Move VAE back
        clean_memory_on_device(device)

        # Prepare Model Input Arguments for FunControl
        y_for_model = y[0] # Shape becomes [32, F, H, W]
        arg_c = {
            "context": context,
            "clip_fea": clip_context,
            "seq_len": seq_len,
            "y": [y_for_model], # Pass the 4D tensor in the list
        }
        arg_null = {
            "context": context_null,
            "clip_fea": clip_context,
            "seq_len": seq_len,
            "y": [y_for_model], # Pass the 4D tensor in the list
        }

        if fun_ref_latent is not None:
            # Model forward expects fun_ref directly, not in a list like 'y'
            arg_c["fun_ref"] = fun_ref_latent
            arg_null["fun_ref"] = fun_ref_latent # Pass to both cond and uncond
            logger.info("Added fun_ref latent to model inputs.")

        # Return noise, context, context_null, y (for potential debugging), (arg_c, arg_null)
        return noise, context, context_null, y, (arg_c, arg_null)

    else:
        # --- Standard I2V Path (Logic copied/adapted from original wan_generate_video.py) ---
        logger.info("Preparing inputs for standard I2V.")

        # get video dimensions
        height, width = args.video_size
        frames = args.video_length # Should be set by now
        max_area = width * height

        # load image (use first image from ref_images for standard I2V)
        if not args.ref_images:
            raise ValueError("--ref_images (single path) is required for standard I2V mode.")
        start_image_path_i2v = args.ref_images.split(',')[0]
        img = Image.open(start_image_path_i2v).convert("RGB")
        img_cv2 = np.array(img)  # PIL to numpy
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device) # For CLIP

        # end frame image
        end_img = None
        end_img_cv2 = None
        if args.end_image_path is not None:
            end_img = Image.open(args.end_image_path).convert("RGB")
            end_img_cv2 = np.array(end_img)  # PIL to numpy
        has_end_image = end_img is not None

        # calculate latent dimensions: keep aspect ratio (Original Method)
        img_height, img_width = img.size[::-1] # PIL size is W,H
        aspect_ratio = img_height / img_width
        lat_h = round(np.sqrt(max_area * aspect_ratio) / config.vae_stride[1] / config.patch_size[1]) * config.patch_size[1]
        lat_w = round(np.sqrt(max_area / aspect_ratio) / config.vae_stride[2] / config.patch_size[2]) * config.patch_size[2]
        target_height = lat_h * config.vae_stride[1]
        target_width = lat_w * config.vae_stride[2]

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #1: Frame Dimension ---
        lat_f_base = (frames - 1) // config.vae_stride[0] + 1  # size of latent frames
        lat_f_effective = lat_f_base + (1 if has_end_image else 0) # Adjust frame dim if end image exists

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #2: Sequence Length ---
        max_seq_len = math.ceil(lat_f_effective * lat_h * lat_w / (config.patch_size[1] * config.patch_size[2]))

        logger.info(f"Standard I2V target pixel resolution: {target_height}x{target_width}, latent shape: ({lat_f_effective}, {lat_h}, {lat_w}), seq_len: {max_seq_len}")

        # set seed
        seed = args.seed
        if not args.cpu_noise:
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(seed)
        else:
            seed_g = torch.manual_seed(seed)

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #3: Noise Shape ---
        noise = torch.randn(
            16, # Channel dim for latent
            lat_f_effective, # Use adjusted frame dim
            lat_h, lat_w,
            dtype=torch.float32, generator=seed_g,
            device=device if not args.cpu_noise else "cpu",
        )
        noise = noise.to(device)

        # configure negative prompt
        n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

        # load text encoder & encode prompts
        text_encoder = load_text_encoder(args, config, device)
        text_encoder.model.to(device)
        with torch.no_grad():
            if args.fp8_t5:
                with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                    context = text_encoder([args.prompt], device)
                    context_null = text_encoder([n_prompt], device)
            else:
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        del text_encoder
        clean_memory_on_device(device)

        # load CLIP model & encode image
        clip = load_clip_model(args, config, device)
        clip.model.to(device)
        logger.info(f"Encoding image to CLIP context")
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
            # Use the [-1, 1] tensor directly if clip.visual expects that format
            # clip_context = clip.visual([img_tensor[:, None, :, :]]).squeeze(1) # Original had [img_tensor[:, None, :, :]] which adds frame dim
            # Use unsqueeze(1) which seems more consistent with other parts
            clip_context = clip.visual([img_tensor.unsqueeze(1)]) # Add Frame dim
        logger.info(f"CLIP Encoding complete")
        del clip
        clean_memory_on_device(device)

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #4: VAE Encoding and 'y' construction ---
        logger.info(f"Encoding image(s) to latent space (Standard I2V method)")
        vae.to_device(device)

        # Resize image(s) for VAE
        interpolation = cv2.INTER_AREA if target_height < img_cv2.shape[0] else cv2.INTER_CUBIC
        img_resized_np = cv2.resize(img_cv2, (target_width, target_height), interpolation=interpolation)
        img_resized = TF.to_tensor(img_resized_np).sub_(0.5).div_(0.5).to(device)  # [-1, 1], CHW
        img_resized = img_resized.unsqueeze(1)  # Add frame dimension -> CFHW, Shape [C, 1, H, W]

        end_img_resized = None
        if has_end_image and end_img_cv2 is not None:
            interpolation_end = cv2.INTER_AREA if target_height < end_img_cv2.shape[0] else cv2.INTER_CUBIC
            end_img_resized_np = cv2.resize(end_img_cv2, (target_width, target_height), interpolation=interpolation_end)
            end_img_resized = TF.to_tensor(end_img_resized_np).sub_(0.5).div_(0.5).to(device) # [-1, 1], CHW
            end_img_resized = end_img_resized.unsqueeze(1) # Add frame dimension -> CFHW, Shape [C, 1, H, W]

        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #5: Mask Shape ---
        msk = torch.zeros(4, lat_f_effective, lat_h, lat_w, device=device, dtype=vae.dtype) # Use adjusted frame dim
        msk[:, 0] = 1 # Mask first frame
        if has_end_image:
            msk[:, -1] = 1 # Mask last frame (the lat_f+1'th frame)

        # Encode image(s) using VAE (Padded Method)
        with accelerator.autocast(), torch.no_grad():
            # Pad the *start* image tensor temporally before encoding
            # Calculate padding needed to reach base frame count (before adding end frame)
            padding_frames_needed = frames - 1 # Number of frames to generate *after* the first
            if padding_frames_needed < 0: padding_frames_needed = 0

            img_padded = img_resized # Start with [C, 1, H, W]
            if padding_frames_needed > 0:
                 # Create padding tensor [C, padding_frames_needed, H, W]
                 padding_tensor = torch.zeros(
                     img_resized.shape[0], padding_frames_needed, img_resized.shape[2], img_resized.shape[3],
                     device=device, dtype=img_resized.dtype
                 )
                 # Concatenate along frame dimension (dim=1)
                 img_padded = torch.cat([img_resized, padding_tensor], dim=1)
                 # Shape should now be [C, 1 + padding_frames_needed, H, W] = [C, frames, H, W]

            # Encode the padded start image tensor. VAE output matches latent frame count.
            # vae.encode expects [C, F, H, W]
            y_latent_base = vae.encode([img_padded])[0] # Shape [C', lat_f_base, H, W]

            if has_end_image and end_img_resized is not None:
                 # Encode the single end frame
                 y_end = vae.encode([end_img_resized])[0] # Shape [C', 1, H, W]
                 # Concatenate along frame dimension (dim=1)
                 y_latent_combined = torch.cat([y_latent_base, y_end], dim=1) # Shape [C', lat_f_base + 1, H, W] = [C', lat_f_effective, H, W]
            else:
                 y_latent_combined = y_latent_base # Shape [C', lat_f_base, H, W] = [C', lat_f_effective, H, W]

        # Concatenate mask and the combined latent
        # --- CRITICAL ORIGINAL LOGIC DIFFERENCE #6: Final 'y' Tensor ---
        y = torch.cat([msk, y_latent_combined], dim=0) # Shape [4+C', lat_f_effective, H, W]
        # y = y.unsqueeze(0) # Add batch dimension? Check model input requirements. Assume model forward handles list/batching.

        logger.info(f"Standard I2V conditioning 'y' constructed. Shape: {y.shape}")
        logger.info(f"Image encoding complete")

        # Move VAE back
        vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
        clean_memory_on_device(device)

        # Prepare model input arguments for Standard I2V
        arg_c = {
            "context": context, # Model expects batch dim? Assuming yes.
            "clip_fea": clip_context,
            "seq_len": max_seq_len, # Use original seq len calculation
            "y": [y], # Use the 'original method' y
        }
        arg_null = {
            "context": context_null,
            "clip_fea": clip_context,
            "seq_len": max_seq_len,
            "y": [y], # Use the 'original method' y
        }

        # Return noise, context, context_null, y (for debugging), (arg_c, arg_null)
        return noise, context, context_null, y, (arg_c, arg_null)
# ========================================================================= #
# END OF MODIFIED FUNCTION prepare_i2v_inputs
# ========================================================================= #

# ========================================================================= #
# START OF NEW FUNCTION prepare_s2v_inputs
# ========================================================================= #
def prepare_s2v_inputs(args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: WanVAE):
    logger.info("Preparing inputs for S2V task.")
    if vae is None:
        raise ValueError("VAE must be provided for S2V input preparation.")

    height, width = args.video_size
    frames_to_generate = args.video_length # Number of frames the DiT should output *excluding* ref frames

    # 1. Load and process reference images
    ref_pil_images = load_ref_images_for_s2v(args.ref_images, (width, height)) # target_size is (W,H)

    # 2. Encode reference images with VAE
    vae.to_device(device)
    ref_latents_list_cond = [] # Will store [C_vae, 1, H_lat, W_lat] tensors
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype):
        for ref_img_pil in ref_pil_images:
            ref_img_tensor = TF.to_tensor(ref_img_pil).sub_(0.5).div_(0.5).to(device) # CHW
            ref_img_tensor = ref_img_tensor.unsqueeze(1) # Add Frame dim: [C,1,H_pix,W_pix]
            # vae.encode expects list of [C,F,H,W], returns list of [C_vae, F_vae, H_lat, W_lat]
            img_vae_latent = vae.encode([ref_img_tensor])[0] # [C_vae, 1, H_lat, W_lat]
            ref_latents_list_cond.append(img_vae_latent)

    # Concatenate along the frame/temporal dimension (dim=1 for [C,F,H,W])
    ref_latents_cond_single_tensor = torch.cat(ref_latents_list_cond, dim=1) # [C_vae, N_refs_latent_frames, H_lat, W_lat]
    ref_latents_cond = ref_latents_cond_single_tensor.unsqueeze(0) # Add Batch: [1, C_vae, N_refs_latent_frames, H_lat, W_lat]
    ref_latents_uncond = torch.zeros_like(ref_latents_cond) # For uncond CFG part

    num_ref_frames_latent = ref_latents_cond.shape[2] # Temporal dim of combined ref latents (F_ref)
    args._s2v_num_ref_frames_latent = num_ref_frames_latent # Store for VAE decoding and sampling loop
    logger.info(f"Encoded {len(ref_pil_images)} ref images. Combined S2V ref_latents shape (batched): {ref_latents_cond.shape}")

    # 3. Calculate target shape for noise (denoisable part + ref part placeholder)
    # Latent frames for the part to be generated by DiT
    video_frames_latent_denoisable = (frames_to_generate - 1) // config.vae_stride[0] + 1 # F_gen_latent
    # Total latent frames for DiT input (generated part + reference part)
    lat_f_combined = video_frames_latent_denoisable + num_ref_frames_latent # F_combined_latent

    lat_h = height // config.vae_stride[1]
    lat_w = width // config.vae_stride[2]

    # DiT input channels should match VAE output channels
    dit_input_channels = getattr(config, 'vae_z_dim', ref_latents_cond.shape[1]) # Use actual VAE z_dim from ref_latents
    noise_shape_with_batch = (1, dit_input_channels, lat_f_combined, lat_h, lat_w)

    # 4. Calculate sequence length for T5 (based on combined latent frames)
    # Patch size for S2V models (from official config structure)
    patch_size_t, patch_size_h, patch_size_w = config.patch_size[0], config.patch_size[1], config.patch_size[2]
    seq_len = math.ceil((lat_h * lat_w) / (patch_size_h * patch_size_w) * (lat_f_combined / patch_size_t) )


    # 5. Set seed and generate noise for the *entire* DiT input shape
    seed = args.seed
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        seed_g = torch.manual_seed(seed)

    noise = torch.randn(noise_shape_with_batch, dtype=torch.float32, generator=seed_g,
                        device=device if not args.cpu_noise else "cpu")
    noise = noise.to(device)
    logger.info(f"S2V initial noise shape (batched): {noise.shape}")


    # 6. Text context
    n_prompt = args.negative_prompt if args.negative_prompt else getattr(config, 'sample_neg_prompt', "")
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device) # Ensure T5 is on device for encoding
    with torch.no_grad():
        context = text_encoder([args.prompt], device)
        context_null = text_encoder([n_prompt], device)
    # Offload T5 if not caching on CPU
    if not getattr(text_encoder, 'fp8', False): # Don't move if T5 is FP8 (assumed on device)
        text_encoder.model.cpu() # Offload T5 after use
    del text_encoder
    clean_memory_on_device(device)

    # 7. Prepare model input args for sampling loop
    arg_c = {"context": context, "seq_len": seq_len, "s2v_ref_latents_cond": ref_latents_cond.to(device)}
    arg_null = {"context": context_null, "seq_len": seq_len, "s2v_ref_latents_uncond": ref_latents_uncond.to(device)}

    # Move VAE back after use
    vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
    clean_memory_on_device(device)

    return noise, context, context_null, (arg_c, arg_null)

# ========================================================================= #
# END OF NEW FUNCTION prepare_s2v_inputs
# ========================================================================= #

# --- V2V Helper Functions ---
# Helper function (copied from generate.py and adapted)
def load_ref_images_for_s2v(paths_str: str, target_size_wh: Tuple[int, int]) -> List[Image.Image]:
    """Load, resize, and pad reference images for S2V task."""
    w, h = target_size_wh # target_size_wh is (width, height)
    ref_paths = paths_str.split(",")
    ref_images_pil = []
    for image_path in ref_paths:
        if not os.path.exists(image_path):
            logger.error(f"Reference image path not found: {image_path}")
            # Consider raising an error or skipping gracefully
            raise FileNotFoundError(f"Reference image path not found: {image_path}")
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_ratio = img.width / img.height
            target_ratio = w / h

            if img_ratio > target_ratio:  # Image is wider than target
                new_width = w; new_height = max(1, int(new_width / img_ratio))
            else:  # Image is taller than target or same ratio
                new_height = h; new_width = max(1, int(new_height * img_ratio))

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            delta_w = w - img_resized.width
            delta_h = h - img_resized.height
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            new_img = ImageOps.expand(img_resized, padding, fill=(255, 255, 255)) # White padding
            ref_images_pil.append(new_img)
    if not ref_images_pil: raise ValueError(f"No valid reference images loaded from: {paths_str}")
    return ref_images_pil

def load_video(video_path, start_frame=0, num_frames=None, bucket_reso=(256, 256)):
    """Load video frames and resize them to the target resolution for V2V.

    Args:
        video_path (str): Path to the video file
        start_frame (int): First frame to load (0-indexed)
        num_frames (int, optional): Number of frames to load. If None, load all frames from start_frame.
        bucket_reso (tuple): Target resolution (height, width)

    Returns:
        list: List of numpy arrays containing video frames in RGB format, resized.
        int: Actual number of frames loaded.
    """
    logger.info(f"Loading video for V2V from {video_path}, target reso {bucket_reso}, frames {start_frame}-{start_frame+num_frames if num_frames else 'end'}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get total frame count and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Input video has {total_frames} frames, {fps} FPS")

    # Calculate how many frames to load
    if num_frames is None:
        frames_to_load = total_frames - start_frame
    else:
        # Make sure we don't try to load more frames than exist
        frames_to_load = min(num_frames, total_frames - start_frame)

    if frames_to_load <= 0:
        cap.release()
        logger.warning(f"No frames to load (start_frame={start_frame}, num_frames={num_frames}, total_frames={total_frames})")
        return [], 0

    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read frames
    frames = []
    target_h, target_w = bucket_reso
    for i in range(frames_to_load):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could only read {len(frames)} frames out of {frames_to_load} requested.")
            break

        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame
        # Use INTER_AREA for downscaling, INTER_LANCZOS4/CUBIC for upscaling
        current_h, current_w = frame_rgb.shape[:2]
        if target_h * target_w < current_h * current_w:
             interpolation = cv2.INTER_AREA
        else:
             interpolation = cv2.INTER_LANCZOS4 # Higher quality for upscaling
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=interpolation)

        frames.append(frame_resized)

    cap.release()
    actual_frames_loaded = len(frames)
    logger.info(f"Successfully loaded and resized {actual_frames_loaded} frames for V2V.")

    return frames, actual_frames_loaded


def encode_video_to_latents(video_tensor: torch.Tensor, vae: WanVAE, device: torch.device, vae_dtype: torch.dtype, args: argparse.Namespace) -> torch.Tensor: # Added args parameter
    """Encode video tensor to latent space using VAE for V2V.

    Args:
        video_tensor (torch.Tensor): Video tensor with shape [B, C, F, H, W], values in [0, 1].
        vae (WanVAE): VAE model instance.
        device (torch.device): Device to perform encoding on.
        vae_dtype (torch.dtype): Target dtype for the output latents.
        args (argparse.Namespace): Command line arguments (needed for vae_cache_cpu). # Added args description

    Returns:
        torch.Tensor: Encoded latents with shape [B, C', F', H', W'].
    """
    if vae is None:
        raise ValueError("VAE must be provided for video encoding.")

    logger.info(f"Encoding video tensor to latents: input shape {video_tensor.shape}")

    # Ensure VAE is on the correct device
    vae.to_device(device)

    # Prepare video tensor: move to device, ensure float32, scale to [-1, 1]
    video_tensor = video_tensor.to(device=device, dtype=torch.float32)
    video_tensor = video_tensor * 2.0 - 1.0 # Scale from [0, 1] to [-1, 1]

    # WanVAE expects input as a list of [C, F, H, W] tensors (no batch dim)
    # Process each video in the batch if batch size > 1 (usually 1 here)
    latents_list = []
    batch_size = video_tensor.shape[0]
    for i in range(batch_size):
        video_single = video_tensor[i] # Shape [C, F, H, W]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=vae.dtype): # Use VAE's internal dtype for autocast
            # vae.encode expects a list containing the tensor
            encoded_latent = vae.encode([video_single])[0] # Returns tensor [C', F', H', W']
            latents_list.append(encoded_latent)

    # Stack results back into a batch
    latents = torch.stack(latents_list, dim=0) # Shape [B, C', F', H', W']

    # Move VAE back to CPU (or cache device)
    # Use the passed args object here
    vae_target_device = torch.device("cpu") if not args.vae_cache_cpu else torch.device("cpu") # Default to CPU, TODO: check if cache device needs specific name
    if args.vae_cache_cpu:
        # Determine the actual cache device if needed, for now, CPU is safe fallback
        logger.info("Moving VAE to CPU for caching (as configured by --vae_cache_cpu).")
    else:
        logger.info("Moving VAE to CPU after encoding.")
    vae.to_device(vae_target_device) # Use args to decide target device
    clean_memory_on_device(device) # Clean the GPU memory

    # Convert latents to the desired final dtype (e.g., bfloat16)
    latents = latents.to(dtype=vae_dtype)
    logger.info(f"Encoded video latents shape: {latents.shape}, dtype: {latents.dtype}")

    return latents


def prepare_v2v_inputs(args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, video_latents: torch.Tensor):
    """Prepare inputs for Video2Video inference based on encoded video latents.

    Args:
        args (argparse.Namespace): Command line arguments.
        config: Model configuration.
        accelerator: Accelerator instance.
        device (torch.device): Device to use.
        video_latents (torch.Tensor): Encoded latent representation of input video [B, C', F', H', W'].

    Returns:
        Tuple containing noise, context, context_null, (arg_c, arg_null).
    """
    # Get dimensions directly from the video latents
    if len(video_latents.shape) != 5:
        raise ValueError(f"Expected video_latents to have 5 dimensions [B, C, F, H, W], but got shape {video_latents.shape}")

    batch_size, latent_channels, lat_f, lat_h, lat_w = video_latents.shape
    if batch_size != 1:
        logger.warning(f"V2V input preparation currently assumes batch size 1, but got {batch_size}. Using first item.")
        video_latents = video_latents[0:1] # Keep batch dim

    target_shape = video_latents.shape[1:] # Get shape without batch dim: [C', F', H', W']

    # Calculate the sequence length based on actual latent dimensions
    patch_h, patch_w = config.patch_size[1], config.patch_size[2]
    spatial_tokens_per_frame = (lat_h * lat_w) // (patch_h * patch_w)
    seq_len = spatial_tokens_per_frame * lat_f
    logger.info(f"V2V derived latent shape: {target_shape}, seq_len: {seq_len}")

    # Configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # Set seed (already set in generate(), just need generator)
    seed = args.seed
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        seed_g = torch.manual_seed(seed)

    # Load text encoder
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)

    # Encode prompt
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)

    # Free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)

    # Generate noise with the same shape as video_latents (including batch dimension)
    noise = torch.randn(
        video_latents.shape, # [B, C', F', H', W']
        dtype=torch.float32,
        device=device if not args.cpu_noise else "cpu",
        generator=seed_g
    )
    noise = noise.to(device) # Ensure noise is on the target device

    # Prepare model input arguments (context needs to match batch size of latents)
    # Assuming batch size 1 for now based on implementation
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    # V2V does not use 'y' or 'clip_fea' in the standard Wan model case
    # If a specific V2V variant *did* need them, they would be added here.

    return noise, context, context_null, (arg_c, arg_null)


# --- End V2V Helper Functions ---

def load_control_video(control_path: str, frames: int, height: int, width: int, args=None) -> torch.Tensor:
    """Load control video to pixel space for Fun-Control model with enhanced control.

    Args:
        control_path: path to control video
        frames: number of frames in the video
        height: height of the video
        width: width of the video
        args: command line arguments (optional, for logging)

    Returns:
        torch.Tensor: control video tensor, CFHW, range [-1, 1]
    """
    logger = logging.getLogger(__name__)
    msg = f"Load control video for Fun-Control from {control_path}"
    if args:
        # Use the correct argument names from wanFUN_generate_video.py
        msg += f" (weight={args.control_weight}, start={args.control_start}, end={args.control_end})"
    logger.info(msg)

    # Use the original helper from hv_generate_video for consistency
    if os.path.isfile(control_path):
        # Use hv_load_video which returns list of numpy arrays (HWC, 0-255)
        video_frames_np = hv_load_video(control_path, 0, frames, bucket_reso=(width, height))
    elif os.path.isdir(control_path):
         # Use hv_load_images which returns list of numpy arrays (HWC, 0-255)
        video_frames_np = hv_load_images(control_path, frames, bucket_reso=(width, height))
    else:
        raise FileNotFoundError(f"Control path not found: {control_path}")

    if not video_frames_np:
         raise ValueError(f"No frames loaded from control path: {control_path}")
    if len(video_frames_np) < frames:
        logger.warning(f"Control video has {len(video_frames_np)} frames, less than requested {frames}. Using available frames.")
        # Optionally, could repeat last frame or loop, but using available is simplest
        frames = len(video_frames_np) # Adjust frame count

    # Stack and convert to tensor: F, H, W, C (0-255) -> F, C, H, W (-1 to 1)
    video_frames_np = np.stack(video_frames_np, axis=0)
    video_tensor = torch.from_numpy(video_frames_np).permute(0, 3, 1, 2).float() / 127.5 - 1.0 # Normalize to [-1, 1]

    # Permute to C, F, H, W
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    logger.info(f"Loaded Fun-Control video tensor shape: {video_tensor.shape}")

    return video_tensor

def setup_scheduler(args: argparse.Namespace, config, device: torch.device) -> Tuple[Any, torch.Tensor]:
    """setup scheduler for sampling

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        Tuple[Any, torch.Tensor]: (scheduler, timesteps)
    """
    if args.sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(args.infer_steps, device=device, shift=args.flow_shift)
        timesteps = scheduler.timesteps
    elif args.sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(args.infer_steps, args.flow_shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
    elif args.sample_solver == "vanilla":
        scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=config.num_train_timesteps, shift=args.flow_shift)
        scheduler.set_timesteps(args.infer_steps, device=device)
        timesteps = scheduler.timesteps

        # FlowMatchDiscreteScheduler does not support generator argument in step method
        org_step = scheduler.step

        def step_wrapper(
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            return_dict: bool = True,
            generator=None, # Add generator argument here
        ):
            # Call original step, ignoring generator if it doesn't accept it
            try:
                # Try calling with generator if the underlying class was updated
                return org_step(model_output, timestep, sample, return_dict=return_dict, generator=generator)
            except TypeError:
                 # Fallback to calling without generator
                 logger.warning("Scheduler step does not support generator argument, proceeding without it.")
                 return org_step(model_output, timestep, sample, return_dict=return_dict)


        scheduler.step = step_wrapper
    else:
        raise NotImplementedError(f"Unsupported solver: {args.sample_solver}")

    logger.info(f"Using scheduler: {args.sample_solver}, timesteps shape: {timesteps.shape}")
    return scheduler, timesteps


def run_sampling(
    model: WanModel,
    noise: torch.Tensor, # This might be pure noise (T2V/I2V) or mixed noise+latent (V2V)
    scheduler: Any,
    timesteps: torch.Tensor, # Might be a subset for V2V
    args: argparse.Namespace,
    inputs: Tuple[dict, dict],
    device: torch.device,
    seed_g: torch.Generator,
    accelerator: Accelerator,
    previewer: Optional[LatentPreviewer] = None, # Add previewer argument
    use_cpu_offload: bool = True, # Example parameter, adjust as needed
    preview_suffix: Optional[str] = None # <<< ADD suffix argument
) -> torch.Tensor:
    """run sampling loop (Denoising)
    Args:
        model: dit model
        noise: initial latent state (pure noise or mixed noise/video latent)
        scheduler: scheduler for sampling
        timesteps: time steps for sampling (can be subset for V2V)
        args: command line arguments
        inputs: model input dictionaries (arg_c, arg_null) containing context etc.
        device: device to use
        seed_g: random generator
        accelerator: Accelerator instance
        previewer: LatentPreviewer instance or None # Added description
        use_cpu_offload: Whether to offload tensors to CPU during processing (example)
        preview_suffix: Unique suffix for preview files to avoid conflicts in concurrent runs.
    Returns:
        torch.Tensor: generated latent
    """
    arg_c, arg_null = inputs

    latent = noise # Initialize latent state
    # Determine storage device (CPU if offloading, otherwise compute device)
    latent_storage_device = torch.device("cpu") if use_cpu_offload else device

    latent = latent.to(latent_storage_device) # Move initial state to storage device

    # cfg skip logic
    apply_cfg_array = []
    num_timesteps = len(timesteps)

    # ... (keep existing cfg skip logic) ...
    if args.cfg_skip_mode != "none" and args.cfg_apply_ratio is not None:
        # Calculate thresholds based on cfg_apply_ratio
        apply_steps = int(num_timesteps * args.cfg_apply_ratio)

        if args.cfg_skip_mode == "early":
            start_index = num_timesteps - apply_steps; end_index = num_timesteps
        elif args.cfg_skip_mode == "late":
            start_index = 0; end_index = apply_steps
        elif args.cfg_skip_mode == "early_late":
            start_index = (num_timesteps - apply_steps) // 2; end_index = start_index + apply_steps
        elif args.cfg_skip_mode == "middle":
            skip_steps = num_timesteps - apply_steps
            middle_start = (num_timesteps - skip_steps) // 2; middle_end = middle_start + skip_steps
        else: # Includes "alternate" - handled inside loop
             start_index = 0; end_index = num_timesteps # Default range for alternate

        w = 0.0 # For alternate mode
        for step_idx in range(num_timesteps):
            apply = True # Default
            if args.cfg_skip_mode == "alternate":
                w += args.cfg_apply_ratio; apply = w >= 1.0
                if apply: w -= 1.0
            elif args.cfg_skip_mode == "middle":
                apply = not (step_idx >= middle_start and step_idx < middle_end)
            elif args.cfg_skip_mode != "none": # early, late, early_late
                apply = step_idx >= start_index and step_idx < end_index

            apply_cfg_array.append(apply)

        pattern = ["A" if apply else "S" for apply in apply_cfg_array]
        pattern = "".join(pattern)
        logger.info(f"CFG skip mode: {args.cfg_skip_mode}, apply ratio: {args.cfg_apply_ratio}, steps: {num_timesteps}, pattern: {pattern}")
    else:
        # Apply CFG on all steps
        apply_cfg_array = [True] * num_timesteps

    # SLG (Skip Layer Guidance) setup
    apply_slg_global = args.slg_layers is not None and args.slg_mode is not None
    slg_start_step = int(args.slg_start * num_timesteps)
    slg_end_step = int(args.slg_end * num_timesteps)

    is_s2v_task_local = "s2v" in args.task


    logger.info(f"Starting sampling loop for {num_timesteps} steps.")
    for i, t in enumerate(tqdm(timesteps)):
        # Prepare input for the model (move latent to compute device)
        # Latent should be [B, C, F, H, W] or [C, F, H, W]
        latent_on_device = latent.to(device)

        # FIX: Check if latent_on_device has too many dimensions and fix it
        # The model expects input x as a list of tensors with shape [C, F, H, W]
        # This adjustment seems specific to a potential bug elsewhere, keep it if needed.
        if len(latent_on_device.shape) > 5:
            while len(latent_on_device.shape) > 5:
                latent_on_device = latent_on_device.squeeze(0)
            logger.debug(f"Adjusted latent shape for model input: {latent_on_device.shape}")

        # The model expects the latent input 'x' as a list: [tensor]
        # If batch dimension is present, we need to split the tensor into a list of tensors
        if len(latent_on_device.shape) == 5:
            # Has batch dimension [B, C, F, H, W]
            latent_model_input_list_template = [latent_on_device[b_idx] for b_idx in range(latent_on_device.shape[0])]
        elif len(latent_on_device.shape) == 4:
            # No batch dimension [C, F, H, W]
            latent_model_input_list_template = [latent_on_device]
        else:
            # Handle unexpected shape
            raise ValueError(f"Latent tensor has unexpected shape {latent_on_device.shape} for model input.")

        timestep = torch.stack([t]).to(device) # Ensure timestep is a tensor on device

        with accelerator.autocast(), torch.no_grad():
            if is_s2v_task_local:
                # S2V specific model input construction and CFG
                # ref_latents_cond/uncond are [B, C_vae, F_ref, H_lat, W_lat]
                ref_c_tensor_batch = arg_c['s2v_ref_latents_cond']
                ref_null_tensor_batch = arg_null['s2v_ref_latents_uncond']
                num_ref_frames_temporal = ref_c_tensor_batch.shape[2] # F_ref from batched tensor

                # latent_on_device is [B, C_dit_in, F_combined, H_lat, W_lat]
                denoisable_part_batch = latent_on_device[:, :, :-num_ref_frames_temporal, :, :] # [B, C, F_gen, H, W]

                # Create input lists for the model (list of [C, F, H, W] tensors)
                # For pos_it and pos_i (conditional image latents)
                input_for_pos_it_and_pos_i_batch = torch.cat([denoisable_part_batch, ref_c_tensor_batch], dim=2)
                x_list_pos_it_and_pos_i = [input_for_pos_it_and_pos_i_batch[b] for b in range(input_for_pos_it_and_pos_i_batch.shape[0])]

                # For neg (unconditional image latents)
                input_for_neg_batch = torch.cat([denoisable_part_batch, ref_null_tensor_batch], dim=2)
                x_list_neg = [input_for_neg_batch[b] for b in range(input_for_neg_batch.shape[0])]

                # Text arguments for each prediction:
                text_args_pos_it = {"context": arg_c["context"], "seq_len": arg_c["seq_len"]}
                text_args_pos_i = {"context": arg_null["context"], "seq_len": arg_null["seq_len"]}
                text_args_neg = {"context": arg_null["context"], "seq_len": arg_null["seq_len"]}


                noise_pred_pos_it = model(x_list_pos_it_and_pos_i, t=timestep, **text_args_pos_it)[0].to(latent_storage_device)
                noise_pred_pos_i = model(x_list_pos_it_and_pos_i, t=timestep, **text_args_pos_i)[0].to(latent_storage_device)
                noise_pred_neg = model(x_list_neg, t=timestep, **text_args_neg)[0].to(latent_storage_device)

                noise_pred = noise_pred_neg + \
                             args.guide_scale_img * (noise_pred_pos_i - noise_pred_neg) + \
                             args.guide_scale_text * (noise_pred_pos_it - noise_pred_pos_i)
            else:
                # Standard CFG / FunControl / I2V / V2V logic
                # Use latent_model_input_list_template which was prepared from latent_on_device
                noise_pred_cond = model(latent_model_input_list_template, t=timestep, **arg_c)[0]
                noise_pred_cond = noise_pred_cond.to(latent_storage_device)

                apply_cfg = apply_cfg_array[i]
                if apply_cfg:
                    apply_slg_step = apply_slg_global and (i >= slg_start_step and i < slg_end_step)
                    slg_indices_for_call = args.slg_layers if apply_slg_step else None
                    uncond_input_args = arg_null

                    if apply_slg_step and args.slg_mode == "original":
                        noise_pred_uncond = model(latent_model_input_list_template, t=timestep, **uncond_input_args)[0].to(latent_storage_device)
                        skip_layer_out = model(latent_model_input_list_template, t=timestep, skip_block_indices=slg_indices_for_call, **uncond_input_args)[0].to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        noise_pred = noise_pred + args.slg_scale * (noise_pred_cond - skip_layer_out)
                    elif apply_slg_step and args.slg_mode == "uncond":
                        noise_pred_uncond = model(latent_model_input_list_template, t=timestep, skip_block_indices=slg_indices_for_call, **uncond_input_args)[0].to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else: # Regular CFG
                        noise_pred_uncond = model(latent_model_input_list_template, t=timestep, **uncond_input_args)[0].to(latent_storage_device)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # CFG is skipped, use conditional prediction directly
                    noise_pred = noise_pred_cond
            # --- End prediction logic ---

            # 3. Compute previous sample state with the scheduler
            # Ensure noise_pred and latent_on_device have matching batch dimensions for scheduler
            # latent_on_device is [B,C,F,H,W]. noise_pred is [C,F,H,W] or [B,C,F,H,W] if batch_size=1 already.
            # Scheduler expects noise_pred [B, C, F, H, W] and sample [B, C, F, H, W]
            if len(noise_pred.shape) == 4 and latent_on_device.shape[0] == 1: # noise_pred is [C,F,H,W]
                 noise_pred_for_scheduler = noise_pred.unsqueeze(0) # Add batch dim: [1,C,F,H,W]
            elif len(noise_pred.shape) == 5 and noise_pred.shape[0] == latent_on_device.shape[0]: # Already batched correctly
                 noise_pred_for_scheduler = noise_pred
            else:
                 raise ValueError(f"Shape mismatch for scheduler: noise_pred {noise_pred.shape}, latent_on_device {latent_on_device.shape}")


            scheduler_output = scheduler.step(
                noise_pred_for_scheduler.to(device), # Ensure noise_pred is on compute device for step
                t,
                latent_on_device, # Pass the tensor (with batch dim) on compute device
                return_dict=False,
                generator=seed_g
            ) # yapf: disable
            prev_latent = scheduler_output[0] # Get the new latent state [B, C, F, H, W]

            # 4. Update latent state (move back to storage device)
            latent = prev_latent.to(latent_storage_device)

            # --- Latent Preview Call ---
            # Preview the state *after* step 'i' is completed
            if previewer is not None and (i + 1) % args.preview == 0 and (i + 1) < num_timesteps:
                 try:
                      logger.debug(f"Generating preview for step {i + 1}")
                      # Pass the *resulting* latent from this step (prev_latent).
                      # Ensure it's on the compute device for the previewer call.
                      # LatentPreviewer handles internal device management.
                      # Need to pass without batch dim if previewer expects [C, F, H, W]
                      # Check LatentPreviewer.preview expects [C, F, H, W]
                      if len(prev_latent.shape) == 5:
                          preview_latent_input = prev_latent.squeeze(0) # Remove batch dim
                      else:
                          preview_latent_input = prev_latent # Assume already [C, F, H, W]

                      # Pass the latent on the main compute device
                      print(f"DEBUG run_sampling: Step {i}, prev_latent shape: {prev_latent.shape}, preview_latent_input shape: {preview_latent_input.shape}")
                      previewer.preview(preview_latent_input.to(device), i, preview_suffix=preview_suffix) # Pass 0-based index 'i'
                 except Exception as e:
                      logger.error(f"Error during latent preview generation at step {i + 1}: {e}", exc_info=True)
                      # Optional: Disable previewer after first error to avoid repeated logs/errors
                      # logger.warning("Disabling latent preview due to error.")
                      # previewer = None

    # Return the final denoised latent (should be on storage device)
    logger.info("Sampling loop finished.")
    return latent

def generate(args: argparse.Namespace) -> Optional[torch.Tensor]:
    """main function for generation pipeline (T2V, I2V, V2V)

    Args:
        args: command line arguments

    Returns:
        Optional[torch.Tensor]: generated latent tensor [B, C, F, H, W], or None if only saving merged model.
    """
    device = torch.device(args.device)
    cfg = WAN_CONFIGS[args.task]

    # --- Determine Mode ---
    is_s2v_task = "s2v" in args.task # Must check S2V first
    is_i2v = args.ref_images is not None and not is_s2v_task # I2V if ref_images given AND not S2V
    is_v2v = args.video_path is not None
    is_fun_control = args.control_path is not None and cfg.is_fun_control # FunControl can overlay I2V/T2V
    is_t2v = not is_i2v and not is_v2v and not is_fun_control and not is_s2v_task


    if is_s2v_task: logger.info(f"Running Subject-to-Video (S2V) inference")
    elif is_v2v: logger.info(f"Running Video-to-Video (V2V) inference with strength {args.strength}")
    elif is_i2v and is_fun_control: logger.info(f"Running Image-to-Video (I2V) with Fun-Control")
    elif is_i2v: logger.info(f"Running Image-to-Video (I2V) inference")
    elif is_fun_control: logger.info(f"Running Text-to-Video with Fun-Control")
    else: logger.info(f"Running Text-to-Video (T2V) inference")


    # --- Data Types ---
    dit_ckpt_path_for_dtype = args.phantom_ckpt if is_s2v_task and args.phantom_ckpt else args.dit
    dit_dtype = detect_wan_sd_dtype(dit_ckpt_path_for_dtype) if dit_ckpt_path_for_dtype is not None else torch.bfloat16

    if dit_dtype.itemsize == 1: # FP8 weights loaded
        dit_dtype = torch.bfloat16 # Use bfloat16 for computation
        if args.fp8_scaled:
            raise ValueError("Cannot use --fp8_scaled with pre-quantized FP8 weights.")
        dit_weight_dtype = None # Weights are already FP8
    elif args.fp8_scaled:
        dit_weight_dtype = None # Optimize later
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn
    else:
        dit_weight_dtype = dit_dtype # Use compute dtype for weights

    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else (torch.bfloat16 if dit_dtype == torch.bfloat16 else torch.float16)
    logger.info(
        f"Using device: {device}, DiT compute: {dit_dtype}, DiT weight: {dit_weight_dtype or 'Mixed (FP8 Scaled)' if args.fp8_scaled else dit_dtype}, VAE: {vae_dtype}, T5 FP8: {args.fp8_t5}"
    )

    # --- Accelerator ---
    mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

    # --- Seed ---
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # Store seed back for metadata
    logger.info(f"Using seed: {seed}")

    # --- Load VAE (if needed for input processing) ---
    vae = None
    # VAE is needed early for V2V, I2V (both types), FunControl T2V, and S2V
    needs_vae_early = is_v2v or is_i2v or (is_fun_control and is_t2v) or \
                      (is_fun_control and is_i2v) or is_s2v_task
    if needs_vae_early:
        vae = load_vae(args, cfg, device, vae_dtype)
        # Keep VAE on specified device for now, will be moved as needed

    # --- Prepare Inputs ---
    noise = None
    context = None
    context_null = None
    inputs = None
    video_latents = None # For V2V mixing

    if is_s2v_task:
        if args.video_length is None:
             raise ValueError("video_length must be specified for S2V mode.")
        noise, context, context_null, inputs = prepare_s2v_inputs(args, cfg, accelerator, device, vae)
    elif is_v2v:
        # Standard V2V path (mutually exclusive with FunControl)
        # 1. Load and prepare video
        video_frames_np, actual_frames_loaded = load_video(
            args.video_path,
            start_frame=0,
            num_frames=args.video_length, # Can be None
            bucket_reso=tuple(args.video_size)
        )
        if actual_frames_loaded == 0:
             raise ValueError(f"Could not load any frames from video: {args.video_path}")

        # Update video_length if it was None or if fewer frames were loaded
        if args.video_length is None or actual_frames_loaded < args.video_length:
            logger.info(f"Updating video_length based on loaded frames: {actual_frames_loaded}")
            args.video_length = actual_frames_loaded
            # Re-check height/width/length now that length is known
            height, width, video_length = check_inputs(args) # This updates args.video_size via side effect if not careful
            args.video_size = [height, width] # Re-store if check_inputs modifies it
        else:
            video_length = args.video_length # Use the specified length

        # Convert frames to tensor [1, C, F, H, W], range [0, 1]
        video_tensor = torch.from_numpy(np.stack(video_frames_np, axis=0)) #[F,H,W,C]
        video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 255.0 #[F,C,H,W]
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0) #[1,C,F,H,W]

        # 2. Encode video to latents
        video_latents = encode_video_to_latents(video_tensor, vae, device, dit_dtype, args) # Use DiT dtype for latents
        del video_tensor # Free pixel video memory
        clean_memory_on_device(device)

        # 3. Prepare V2V inputs (noise matching latent shape, context, etc.)
        noise, context, context_null, inputs = prepare_v2v_inputs(args, cfg, accelerator, device, video_latents)

    elif is_i2v: # This now covers both standard I2V and FunControl-I2V
        if args.video_length is None:
             raise ValueError("video_length must be specified for I2V mode.")
        noise, context, context_null, _, inputs = prepare_i2v_inputs(args, cfg, accelerator, device, vae)
        # Note: prepare_i2v_inputs moves VAE to CPU/cache after use

    elif is_fun_control: # Pure FunControl T2V (no image input unless using start/end image)
        if args.video_length is None:
             raise ValueError("video_length must be specified for Fun-Control T2V mode.")
        noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, vae)
        # Note: prepare_t2v_inputs moves VAE to CPU/cache if it used it

    elif is_t2v: # Standard T2V
        if args.video_length is None:
             raise ValueError("video_length must be specified for standard T2V mode.")
        noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, None) # Pass None for VAE


    # At this point, VAE should be on CPU/cache unless still needed for decoding
    # If VAE wasn't loaded early (standard T2V), vae is still None

    # --- Load DiT Model ---
    model = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v) # Pass is_i2v flag

    # --- Merge LoRA ---
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        merge_lora_weights(model, args, device)
        if args.save_merged_model:
            logger.info("Merged model saved. Exiting without generation.")
            # Clean up resources if exiting early
            if 'model' in locals(): del model
            if 'vae' in locals() and vae is not None: del vae
            clean_memory_on_device(device)
            return None # Exit early

    # --- Optimize Model (FP8, Swapping, Compile) ---
    optimize_model(model, args, device, dit_dtype, dit_weight_dtype)

    # --- Setup Scheduler & Timesteps ---
    scheduler, timesteps = setup_scheduler(args, cfg, device)

    # --- Prepare for Sampling ---
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # `latent` here is the initial state *before* the sampling loop starts
    latent = noise # Start with noise (already shaped correctly for T2V/I2V/V2V/S2V)

    # --- V2V Strength Adjustment ---
    if is_v2v and args.strength < 1.0: # Not applicable to S2V
        if video_latents is None:
             raise RuntimeError("video_latents not available for V2V strength adjustment.")

        # Calculate number of inference steps based on strength
        num_inference_steps = max(1, int(args.infer_steps * args.strength))
        logger.info(f"V2V Strength: {args.strength}, adjusting inference steps from {args.infer_steps} to {num_inference_steps}")

        # Get starting timestep index and value
        t_start_idx = len(timesteps) - num_inference_steps
        if t_start_idx < 0: t_start_idx = 0 # Ensure non-negative index
        t_start = timesteps[t_start_idx] # Timestep value at the start of sampling

        # Mix noise and video latents based on starting timestep using scheduler
        # Ensure video_latents are on the same device and dtype as noise for mixing
        video_latents = video_latents.to(device=latent.device, dtype=latent.dtype)

        if latent.shape != video_latents.shape:
            logger.error(f"Noise shape {latent.shape} does not match video latent shape {video_latents.shape} for V2V mixing. Cannot proceed.")
            raise ValueError("Shape mismatch between noise and video latents in V2V.")

        # Use scheduler's add_noise for better mixing
        latent = scheduler.add_noise(video_latents, latent, t_start.unsqueeze(0))
        logger.info(f"Mixed video latents and noise using scheduler at timestep {t_start.item():.1f}")

        # Use only the required subset of timesteps
        timesteps = timesteps[t_start_idx:]
        logger.info(f"Using last {len(timesteps)} timesteps for V2V sampling.")
    else:
         logger.info(f"Using full {len(timesteps)} timesteps for sampling.")
         # Latent remains the initial noise (or S2V prepared noise)

    # --- Initialize Latent Previewer --- # ADDED SECTION
    previewer = None
    if LatentPreviewer is not None and args.preview is not None and args.preview > 0:
        logger.info(f"Initializing Latent Previewer (every {args.preview} steps)...")
        try:
             # Use the initial 'latent' state which might be pure noise or mixed V2V start
             # Pass without batch dim [C, F, H, W]
             # For S2V, preview the denoisable part only for meaningful previews
             if is_s2v_task and hasattr(args, "_s2v_num_ref_frames_latent"):
                 num_ref_frames_to_trim_preview = args._s2v_num_ref_frames_latent
                 initial_latent_for_preview = latent.clone().squeeze(0)[:, :-num_ref_frames_to_trim_preview, :, :]
             else:
                 initial_latent_for_preview = latent.clone().squeeze(0)
             previewer = LatentPreviewer(args, initial_latent_for_preview, timesteps, device, dit_dtype, model_type="wan")
             logger.info("Latent Previewer initialized successfully.")
        except Exception as e:
             logger.error(f"Failed to initialize Latent Previewer: {e}", exc_info=True)
             previewer = None # Ensure it's None if init fails
    # --- END ADDED SECTION ---

    # --- Run Sampling Loop ---
    logger.info("Starting denoising sampling loop...")
    final_latent = run_sampling(
        model,
        latent, # Initial state (noise or mixed)
        scheduler,
        timesteps, # Full or partial timesteps
        args,
        inputs, # Contains context etc.
        device,
        seed_g,
        accelerator,
        previewer=previewer, # MODIFIED: Pass the previewer instance
        use_cpu_offload=(args.blocks_to_swap > 0), # Example: offload if swapping
        preview_suffix=args.preview_suffix # <<< Pass the suffix from args
    )

    # --- Cleanup ---
    del model
    if 'scheduler' in locals(): del scheduler
    if 'context' in locals(): del context
    if 'context_null' in locals(): del context_null
    if 'inputs' in locals(): del inputs # Free memory from encoded inputs
    if video_latents is not None: del video_latents
    # previewer instance will be garbage collected

    synchronize_device(device)

    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to ensure block swap finishes...")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    # Store VAE instance in args for decoding function (if it exists)
    args._vae = vae # Store VAE instance (might be None if T2V)

    # Return latent with batch dimension [1, C, F, H, W]
    # final_latent is potentially on CPU if use_cpu_offload=True
    if len(final_latent.shape) == 4: # If run_sampling returned [C, F, H, W]
        final_latent = final_latent.unsqueeze(0)

    return final_latent

def decode_latent(latent: torch.Tensor, args: argparse.Namespace, cfg) -> torch.Tensor:
    """decode latent tensor to video frames

    Args:
        latent: latent tensor [B, C, F, H, W]
        args: command line arguments (contains _vae instance)
        cfg: model configuration

    Returns:
        torch.Tensor: decoded video tensor [B, C, F, H, W], range [0, 1], on CPU
    """
    device = torch.device(args.device)

    # Load VAE model or use the one from the generation pipeline
    vae = None
    if hasattr(args, "_vae") and args._vae is not None:
        vae = args._vae
        logger.info("Using VAE instance from generation pipeline for decoding.")
    else:
        # Need to load VAE if it wasn't used/stored (e.g., pure T2V or latent input mode)
        logger.info("Loading VAE for decoding...")
        # Attempt to detect DiT dtype even if DiT wasn't loaded (e.g., latent mode)
        # Fallback to bfloat16 if DiT path isn't available
        try:
            dit_ckpt_path_for_dtype_decode = args.phantom_ckpt if "s2v" in args.task and args.phantom_ckpt else args.dit
            dit_dtype_ref = detect_wan_sd_dtype(dit_ckpt_path_for_dtype_decode) if dit_ckpt_path_for_dtype_decode else torch.bfloat16
        except: # Handle cases where DiT path is invalid or missing in latent mode
            dit_dtype_ref = torch.bfloat16
            logger.warning("Could not detect DiT dtype for VAE decoding, defaulting to bfloat16.")

        vae_dtype_decode = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else (torch.bfloat16 if dit_dtype_ref == torch.bfloat16 else torch.float16)
        vae = load_vae(args, cfg, device, vae_dtype_decode)
        args._vae = vae # Store it in case needed again?

    # Ensure VAE is on device for decoding
    vae.to_device(device)

    logger.info(f"Decoding video from latents: shape {latent.shape}, dtype {latent.dtype}")
    
    latent_to_decode_vae = latent
    if "s2v" in args.task and hasattr(args, "_s2v_num_ref_frames_latent"):
        num_ref_frames_to_trim = args._s2v_num_ref_frames_latent
        if num_ref_frames_to_trim > 0 and latent.shape[2] > num_ref_frames_to_trim : # Ensure temporal dim is large enough
            logger.info(f"S2V mode: Trimming reference latent part ({num_ref_frames_to_trim} frames) before VAE decoding.")
            # Trim from the end of the temporal dimension (dim=2 for [B,C,F,H,W])
            latent_to_decode_vae = latent[:, :, :-num_ref_frames_to_trim, :, :]
            logger.info(f"Latent shape for VAE decoding (S2V trimmed): {latent_to_decode_vae.shape}")


    # Ensure latent is on the correct device and expected dtype for VAE
    latent_decode_on_device = latent_to_decode_vae.to(device=device, dtype=vae.dtype)


    # VAE decode expects list of [C, F, H, W] or a single [B, C, F, H, W]
    # WanVAE wrapper seems to handle the list internally now? Check its decode method.
    # Assuming it takes [B, C, F, H, W] directly or handles the list internally.
    videos = None
    with torch.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
        # WanVAE.decode returns a list of decoded videos [C, F, H, W]
        decoded_list = vae.decode(latent_decode_on_device) # Pass the batch tensor
        if decoded_list and len(decoded_list) > 0:
             # Stack list back into batch dimension: B, C, F, H, W
             videos = torch.stack(decoded_list, dim=0)
        else:
             raise RuntimeError("VAE decoding failed or returned empty list.")


    # Move VAE back to CPU/cache
    vae.to_device(args.vae_cache_cpu if args.vae_cache_cpu else "cpu")
    clean_memory_on_device(device)

    logger.info(f"Decoded video shape: {videos.shape}")

    # Post-processing: trim tail frames, convert to float32 CPU, scale to [0, 1]
    if args.trim_tail_frames > 0:
        logger.info(f"Trimming last {args.trim_tail_frames} frames.")
        videos = videos[:, :, : -args.trim_tail_frames, :, :]

    # Scale from [-1, 1] (VAE output range) to [0, 1] (video save range)
    videos = (videos + 1.0) / 2.0
    videos = torch.clamp(videos, 0.0, 1.0)

    # Move to CPU and convert to float32 for saving
    video_final = videos.cpu().to(torch.float32)
    logger.info(f"Decoding complete. Final video tensor shape: {video_final.shape}")

    return video_final


def save_output(
    video_tensor: torch.Tensor, # Expects [B, C, F, H, W] range [0, 1]
    args: argparse.Namespace,
    original_base_names: Optional[List[str]] = None,
    latent_to_save: Optional[torch.Tensor] = None # Optional latent [B, C, F, H, W]
) -> None:
    """save output video, images, or latent

    Args:
        video_tensor: decoded video tensor [B, C, F, H, W], range [0, 1]
        args: command line arguments
        original_base_names: original base names (if latents are loaded from files)
        latent_to_save: optional raw latent tensor to save
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    # Get dimensions from the *decoded* video tensor
    batch_size, channels, video_length, height, width = video_tensor.shape

    base_name = f"{time_flag}_{seed}"
    if original_base_names:
         # Use first original name if loading multiple latents (though currently unsupported)
         base_name += f"_{original_base_names[0]}"

    # --- Save Latent ---
    if (args.output_type == "latent" or args.output_type == "both") and latent_to_save is not None:
        latent_path = os.path.join(save_path, f"{base_name}_latent.safetensors")
        logger.info(f"Saving latent tensor shape: {latent_to_save.shape}")

        metadata = {}
        if not args.no_metadata:
            # Try to get model paths robustly for metadata
            cfg = WAN_CONFIGS.get(args.task) # Get config if task exists
            dit_path_meta = "N/A"
            if "s2v" in args.task and args.phantom_ckpt:
                dit_path_meta = args.phantom_ckpt
            elif args.dit: dit_path_meta = args.dit
            elif cfg and cfg.dit_checkpoint and args.ckpt_dir: dit_path_meta = os.path.join(args.ckpt_dir, cfg.dit_checkpoint)
            elif cfg and cfg.dit_checkpoint: dit_path_meta = cfg.dit_checkpoint # Use relative path if no ckpt_dir


            vae_path_meta = "N/A"
            if args.vae: vae_path_meta = args.vae
            elif cfg and cfg.vae_checkpoint and args.ckpt_dir: vae_path_meta = os.path.join(args.ckpt_dir, cfg.vae_checkpoint)
            elif cfg and cfg.vae_checkpoint: vae_path_meta = cfg.vae_checkpoint # Use relative path if no ckpt_dir

            mode_string = "T2V"
            if "s2v" in args.task: mode_string = "S2V"
            elif args.video_path: mode_string = "V2V"
            elif args.ref_images and args.control_path: mode_string = "FunControl-I2V"
            elif args.control_path: mode_string = "FunControl-T2V"
            elif args.ref_images: mode_string = "I2V"


            metadata = {
                "prompt": f"{args.prompt}",
                "negative_prompt": f"{args.negative_prompt or ''}",
                "seeds": f"{seed}",
                "height": f"{height}", # Use decoded height/width
                "width": f"{width}",
                "video_length": f"{video_length}", # Use decoded length
                "infer_steps": f"{args.infer_steps}",
                "guidance_scale": f"{args.guidance_scale}", # General CFG
                "flow_shift": f"{args.flow_shift}",
                "task": f"{args.task}",
                "dit_model": f"{dit_path_meta}",
                "vae_model": f"{vae_path_meta}",
                "mode": mode_string,
            }
            if args.video_path: metadata["v2v_strength"] = f"{args.strength}" # yapf: disable
            if args.ref_images: metadata["ref_images"] = f"{args.ref_images}" # Store all ref_images paths
            if args.end_image_path: metadata["i2v_end_image"] = f"{os.path.basename(args.end_image_path)}"
            if args.control_path:
                 metadata["funcontrol_video"] = f"{os.path.basename(args.control_path)}"
                 metadata["funcontrol_weight"] = f"{args.control_weight}"
                 metadata["funcontrol_start"] = f"{args.control_start}"
                 metadata["funcontrol_end"] = f"{args.control_end}"
                 metadata["funcontrol_falloff"] = f"{args.control_falloff_percentage}"
            if "s2v" in args.task:
                # metadata["phantom_ckpt"] = f"{os.path.basename(args.phantom_ckpt) if args.phantom_ckpt else 'N/A'}" # Already in dit_model
                metadata["guide_scale_img"] = f"{args.guide_scale_img}"
                metadata["guide_scale_text"] = f"{args.guide_scale_text}"

            # Add LoRA info if used
            if args.lora_weight:
                metadata["lora_weights"] = ", ".join([os.path.basename(p) for p in args.lora_weight])
                metadata["lora_multipliers"] = ", ".join(map(str, args.lora_multiplier))


        # Ensure latent is on CPU for saving
        sd = {"latent": latent_to_save.cpu()}
        try:
            save_file(sd, latent_path, metadata=metadata)
            logger.info(f"Latent saved to: {latent_path}")
        except Exception as e:
            logger.error(f"Failed to save latent file: {e}")


    # --- Save Video or Images ---
    if args.output_type == "video" or args.output_type == "both":
        video_path = os.path.join(save_path, f"{base_name}.mp4")
        # save_videos_grid expects [B, T, H, W, C], need to permute and rescale if needed
        # Input video_tensor is [B, C, T, H, W], range [0, 1]
        # save_videos_grid handles the rescale flag correctly if input is [0,1]
        try:
            save_videos_grid(video_tensor, video_path, fps=args.fps, rescale=False) # Pass rescale=False as tensor is already [0,1]
            logger.info(f"Video saved to: {video_path}")
        except Exception as e:
            logger.error(f"Failed to save video file: {e}")
            logger.error(f"Video tensor info: shape={video_tensor.shape}, dtype={video_tensor.dtype}, min={video_tensor.min()}, max={video_tensor.max()}")


    elif args.output_type == "images":
        image_save_dir = os.path.join(save_path, base_name)
        os.makedirs(image_save_dir, exist_ok=True)
        # save_images_grid expects [B, T, H, W, C], need to permute and rescale if needed
        # Input video_tensor is [B, C, T, H, W], range [0, 1]
        # save_images_grid handles the rescale flag correctly if input is [0,1]
        try:
             # Save as individual frames
             save_images_grid(video_tensor, image_save_dir, "frame", rescale=False, save_individually=True) # Pass rescale=False
             logger.info(f"Image frames saved to directory: {image_save_dir}")
        except Exception as e:
            logger.error(f"Failed to save image files: {e}")


def main():
    # --- Argument Parsing & Setup ---
    args = parse_args()

    # Determine mode: generation or loading latents
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device_str = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device(device_str) # Store device back in args
    logger.info(f"Using device: {args.device}")

    generated_latent = None # To hold the generated latent if not in latents_mode
    cfg = WAN_CONFIGS[args.task] # Get config early for potential use
    height, width, video_length = None, None, None # Initialize dimensions
    original_base_names = None # For naming output when loading latents

    if not latents_mode:
        # --- Generation Mode (T2V, I2V, V2V, Fun-Control, S2V) ---
        logger.info("Running in Generation Mode")
        # Setup arguments (defaults, etc.)
        args = setup_args(args)
        # Validate inputs (initial check, V2V might refine length later)
        height, width, video_length = check_inputs(args)
        args.video_size = [height, width] # Ensure args reflect checked dimensions
        args.video_length = video_length # May still be None for V2V

        # Determine specific mode string
        mode_str = "Unknown"
        if "s2v" in args.task: mode_str = "S2V"
        elif args.video_path: mode_str = "V2V"
        elif args.ref_images and args.control_path: mode_str = "FunControl-I2V" # FunControl overrides if control_path is present
        elif args.control_path: mode_str = "FunControl-T2V"
        elif args.ref_images: mode_str = "I2V"
        else: mode_str = "T2V"



        logger.info(f"Mode: {mode_str} (Task: {args.task})")
        logger.info(
            f"Initial settings: video size: {height}x{width}@{video_length or 'auto'} (HxW@F), fps: {args.fps}, "
            f"infer_steps: {args.infer_steps}, flow_shift: {args.flow_shift}"
        )
        if mode_str == "V2V": logger.info(f"V2V Strength: {args.strength}")
        if "FunControl" in mode_str: logger.info(f"FunControl Weight: {args.control_weight}, Start: {args.control_start}, End: {args.control_end}, Falloff: {args.control_falloff_percentage}")
        if mode_str == "S2V": logger.info(f"S2V: Phantom Ckpt: {args.phantom_ckpt}, Img Scale: {args.guide_scale_img}, Text Scale: {args.guide_scale_text}, Guidance Scale (general): {args.guidance_scale} (Not used by S2V CFG)")
        else: logger.info(f"Guidance Scale: {args.guidance_scale}")


        # Core generation pipeline
        generated_latent = generate(args) # Returns [B, C, F, H, W] or None

        if args.save_merged_model:
            logger.info("Exiting after saving merged model.")
            return # Exit if only saving model

        if generated_latent is None:
             logger.error("Generation failed or was skipped, exiting.")
             return

        # Update dimensions based on the *actual* generated latent
        # Latent shape might differ slightly from input request depending on VAE/model strides
        _, lat_c, lat_f_final_latent, lat_h, lat_w = generated_latent.shape
        
        # For S2V, lat_f_final_latent includes reference frames.
        # The actual *generated* video frames will be fewer.
        # This is handled in decode_latent for S2V.
        # For metadata, we want the dimensions of the *output video*.
        
        pixel_height = lat_h * cfg.vae_stride[1]
        pixel_width = lat_w * cfg.vae_stride[2]

        if "s2v" in args.task and hasattr(args, "_s2v_num_ref_frames_latent"):
            num_ref_frames_latent_metadata = args._s2v_num_ref_frames_latent
            # The frames in the *output video* are from the denoisable part of the latent
            pixel_frames = (lat_f_final_latent - num_ref_frames_latent_metadata -1) * cfg.vae_stride[0] + 1 if lat_f_final_latent > num_ref_frames_latent_metadata else 1
        else:
            pixel_frames = (lat_f_final_latent - 1) * cfg.vae_stride[0] + 1

        logger.info(f"Generation complete. Final latent shape (pre-decode processing): {generated_latent.shape} -> Approx. Pixel Video: {pixel_height}x{pixel_width}@{pixel_frames}")
        # Use these derived pixel dimensions for saving metadata if they are more accurate
        # However, save_output will use the dimensions of the *decoded_video* tensor
        # So, we only update these for logging here. Actual dimensions for metadata
        # will come from the decoded video in save_output.
        # height, width, video_length = pixel_height, pixel_width, pixel_frames


    else:
        # --- Latents Mode (Load and Decode) ---
        logger.info("Running in Latent Loading Mode")
        original_base_names = []
        latents_list = []
        seeds = [] # Try to recover seed from metadata

        # Currently only supporting one latent file input
        if len(args.latent_path) > 1:
            logger.warning("Loading multiple latent files is not fully supported for metadata merging. Using first file's info.")

        latent_path = args.latent_path[0]
        original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
        loaded_latent = None
        metadata = {}
        seed = args.seed if args.seed is not None else random.randint(0, 2**32-1) # Default seed if none in metadata

        try:
            if os.path.splitext(latent_path)[1].lower() != ".safetensors":
                logger.warning("Loading non-safetensors latent file. Metadata might be missing.")
                loaded_latent = torch.load(latent_path, map_location="cpu")
                # Attempt to handle different save formats (dict vs raw tensor)
                if isinstance(loaded_latent, dict):
                    if "latent" in loaded_latent:
                        loaded_latent = loaded_latent["latent"]
                    elif "state_dict" in loaded_latent: # Might be a full model checkpoint by mistake
                         raise ValueError("Loaded file appears to be a model checkpoint, not a latent tensor.")
                    else: # Try the first value if it's a tensor
                         first_key = next(iter(loaded_latent))
                         if isinstance(loaded_latent[first_key], torch.Tensor):
                              loaded_latent = loaded_latent[first_key]
                         else:
                              raise ValueError("Could not find latent tensor in loaded dictionary.")
                elif not isinstance(loaded_latent, torch.Tensor):
                     raise ValueError(f"Loaded file content is not a tensor or expected dictionary format: {type(loaded_latent)}")


            else:
                # Load latent tensor
                loaded_latent_data = st_load_file(latent_path, device="cpu")
                if "latent" not in loaded_latent_data:
                    raise ValueError(f"'latent' key not found in safetensors file: {latent_path}")
                loaded_latent = loaded_latent_data["latent"]

                # Load metadata
                with safe_open(latent_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                logger.info(f"Loaded metadata: {metadata}")

                # Restore args from metadata if available AND not overridden by command line
                # Command line args take precedence if provided
                if args.seed is None and "seeds" in metadata: seed = int(metadata["seeds"])
                if args.prompt == "" and "prompt" in metadata: args.prompt = metadata["prompt"] # Allow empty prompt override from metadata
                elif "prompt" in metadata and not args.prompt: args.prompt = metadata["prompt"]
                
                if args.negative_prompt is None and "negative_prompt" in metadata: args.negative_prompt = metadata["negative_prompt"]

                # We need height/width/length to decode, so always load if available
                # These will be used if args.video_size/length are not set by user for latent mode
                if "height" in metadata: height = int(metadata["height"])
                if "width" in metadata: width = int(metadata["width"])
                if "video_length" in metadata: video_length = int(metadata["video_length"])

                if args.guidance_scale == 5.0 and "guidance_scale" in metadata: args.guidance_scale = float(metadata["guidance_scale"]) # Assuming 5.0 is default
                if args.infer_steps is None and "infer_steps" in metadata: args.infer_steps = int(metadata["infer_steps"])
                if args.flow_shift is None and "flow_shift" in metadata: args.flow_shift = float(metadata["flow_shift"])
                if "task" in metadata: args.task = args.task or metadata["task"] # Restore task if not specified
                
                # S2V specific from metadata
                if "s2v" in args.task: # Only if task itself indicates S2V
                    if args.guide_scale_img == 5.0 and "guide_scale_img" in metadata: args.guide_scale_img = float(metadata["guide_scale_img"])
                    if args.guide_scale_text == 7.5 and "guide_scale_text" in metadata: args.guide_scale_text = float(metadata["guide_scale_text"])
                    if args.phantom_ckpt is None and "dit_model" in metadata and "Phantom" in metadata["dit_model"]:
                        # Try to infer phantom_ckpt if it was saved in metadata's dit_model path
                        # This is a bit heuristic.
                        logger.info(f"Attempting to infer phantom_ckpt from metadata: {metadata['dit_model']}")
                        args.phantom_ckpt = metadata["dit_model"]


                # FunControl specific args
                if "funcontrol_weight" in metadata: args.control_weight = args.control_weight or float(metadata["funcontrol_weight"])
                if "funcontrol_start" in metadata: args.control_start = args.control_start or float(metadata["funcontrol_start"])
                if "funcontrol_end" in metadata: args.control_end = args.control_end or float(metadata["funcontrol_end"])
                if "funcontrol_falloff" in metadata: args.control_falloff_percentage = args.control_falloff_percentage or float(metadata["funcontrol_falloff"])
                # V2V specific args
                if "v2v_strength" in metadata: args.strength = args.strength or float(metadata["v2v_strength"])

                # Update config based on restored task
                cfg = WAN_CONFIGS[args.task]

            seeds.append(seed)
            latents_list.append(loaded_latent)
            logger.info(f"Loaded latent from {latent_path}. Shape: {loaded_latent.shape}, dtype: {loaded_latent.dtype}")

        except Exception as e:
            logger.error(f"Failed to load latent file {latent_path}: {e}")
            return

        if not latents_list:
            logger.error("No latent tensors were loaded.")
            return

        # Stack latents (currently just one) - ensure batch dimension
        generated_latent = torch.stack(latents_list, dim=0) # [B, C, F, H, W]
        if len(generated_latent.shape) != 5:
             # Maybe saved without batch dim? Try adding it.
             if len(generated_latent.shape) == 4:
                 logger.warning(f"Loaded latent has 4 dimensions {generated_latent.shape}. Adding batch dimension.")
                 generated_latent = generated_latent.unsqueeze(0)
             else:
                 raise ValueError(f"Loaded latent has incorrect shape: {generated_latent.shape}. Expected 4 or 5 dimensions.")

        # Set seed from metadata (or default)
        args.seed = seeds[0]

        # Infer pixel dimensions from latent shape and config if not available in metadata
        # OR if user did not provide video_size for latent mode
        _, _, lat_f_loaded, lat_h_loaded, lat_w_loaded = generated_latent.shape # Get dimensions from loaded latent
        
        user_provided_height = args.video_size[0] != 256 or args.video_size[1] != 256 # Check if user changed default
        user_provided_length = args.video_length is not None

        # Use metadata dimensions if user hasn't overridden them for latent mode
        if height is not None and not user_provided_height : args.video_size[0] = height
        if width is not None and not user_provided_height : args.video_size[1] = width
        if video_length is not None and not user_provided_length: args.video_length = video_length


        # If still no valid dimensions, infer from latent
        if args.video_size[0] == 256 and args.video_size[1] == 256 and (height is None or width is None): # Still default and no metadata
             logger.warning("Dimensions not found in metadata and not set by user, inferring from latent shape.")
             args.video_size[0] = lat_h_loaded * cfg.vae_stride[1]
             args.video_size[1] = lat_w_loaded * cfg.vae_stride[2]
             logger.info(f"Inferred pixel dimensions: {args.video_size[0]}x{args.video_size[1]}")
        
        if args.video_length is None and video_length is None: # Still None and no metadata
            if "s2v" in args.task and hasattr(args, "_s2v_num_ref_frames_latent"):
                num_ref_frames_for_length_calc = args._s2v_num_ref_frames_latent # Needs to be set if loading S2V latent
                # This might be tricky if _s2v_num_ref_frames_latent isn't in metadata for S2V latents
                # For now, assume it's mostly for generated latents.
                # A better way would be to save num_ref_frames_latent in S2V latent metadata.
                # As a fallback, if not available, we assume the full latent length is for video.
                if num_ref_frames_for_length_calc is None:
                    logger.warning("S2V latent loaded, but _s2v_num_ref_frames_latent not available. Inferred length might be inaccurate.")
                    num_ref_frames_for_length_calc = 0
                args.video_length = (lat_f_loaded - num_ref_frames_for_length_calc - 1) * cfg.vae_stride[0] + 1 if lat_f_loaded > num_ref_frames_for_length_calc else 1
            else:
                args.video_length = (lat_f_loaded - 1) * cfg.vae_stride[0] + 1
            logger.info(f"Inferred video length: {args.video_length}")


    # --- Decode and Save ---
    if generated_latent is not None:
        # Decode latent to video tensor [B, C, F, H, W], range [0, 1]
        decoded_video = decode_latent(generated_latent, args, cfg)

        # Save the output (latent and/or video/images)
        save_output(
            decoded_video,
            args,
            original_base_names=original_base_names,
            latent_to_save=generated_latent if (args.output_type == "latent" or args.output_type == "both") else None
        )
    else:
        logger.error("No latent available for decoding and saving.")

    logger.info("Done!")


if __name__ == "__main__":
    main()


