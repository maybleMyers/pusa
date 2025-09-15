#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 15:11:58 2025

@author: blyss
"""
import sys
import os
import argparse
import torch
from rich.traceback import install as install_rich_tracebacks
from blissful_tuner.utils import BlissfulLogger, string_to_seed, parse_scheduled_cfg, error_out
logger = BlissfulLogger(__name__, "#8e00ed")

BLISSFUL_VERSION = "0.4.0"

CFG_SCHEDULE_HELP = """
Comma-separated list of steps/ranges where CFG should be applied.

You can specify:
- Single steps (e.g., '5')
- Ranges (e.g., '1-10')
- Modulus patterns (e.g., 'e~2' for every 2 steps)
- Guidance scale overrides (e.g., '1-10:5.0')

Example schedule:
  'e~2:6.4, 1-10, 46-50'

This would apply:
- Default CFG scale for steps 1-10 and 46-50
- 6.4 CFG scale every 2 steps outside that range
- No CFG otherwise

You can exclude steps using '!', e.g., '!32' skips step 32.
Note: The list is processed left to right, so modulus ranges should come first and exclusions at the end!
"""

ROOT_SCRIPT = os.path.basename(sys.argv[0]).lower()
if "hv_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "hunyuan"
elif "wan_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "wan"
elif "fpack_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "framepack"
else:
    raise ValueError("Unsupported root_script for Blissful Extension")

if "generate" in ROOT_SCRIPT:
    MODE = "generate"
elif "train" in ROOT_SCRIPT:
    MODE = "train"
else:
    raise ValueError("Unsupported root script for Blissful Extension!")


def blissful_prefunc(args: argparse.Namespace):
    """Simple function to print about version, environment, and things"""
    cuda_list = [f"PyTorch: {torch.__version__}"]
    if torch.cuda.is_available():
        allocator = torch.cuda.get_allocator_backend()
        cuda = torch.cuda.get_device_properties(0)
        cuda_list[0] += f", CUDA: {torch.version.cuda} CC: {cuda.major}.{cuda.minor}"
        cuda_list.append(f"Device: '{cuda.name}', VRAM: '{cuda.total_memory // 1024 ** 2}MB'")
    for string in cuda_list:
        logger.info(string)
    if args.fp16_accumulation and MODE == "generate":
        logger.info("Enabling FP16 accumulation")
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        else:
            raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation is not available in this version of torch, requires torch 2.7.0.dev2025 02 26 nightly minimum")


def add_blissful_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    install_rich_tracebacks()
    if DIFFUSION_MODEL == "wan":
        parser.add_argument("--noise_aug_strength", type=float, default=0.0, help="Additional multiplier for i2v noise, higher might help motion/quality")
        parser.add_argument("--prompt_weighting", action="store_true", help="Enable (prompt weighting:1.2)")
        parser.add_argument(
            "--rope_func", type=str, default="default",
            help="Function to use for ROPE. Choose from 'default' or 'comfy' the latter of which uses ComfyUI implementation and is compilable with torch.compile to enable BIG VRAM savings"
        )

    elif DIFFUSION_MODEL == "hunyuan":
        parser.add_argument("--hidden_state_skip_layer", type=int, default=2, help="Hidden state skip layer for LLM. Default is 2. Think 'clip skip' for the LLM")
        parser.add_argument("--apply_final_norm", type=bool, default=False, help="Apply final norm for LLM. Default is False. Usually makes things worse.")
        parser.add_argument("--reproduce", action="store_true", help="Enable reproducible output(Same seed = same result. Default is False.")
        parser.add_argument("--fp8_scaled", action="store_true", help="Scaled FP8 quantization. Better quality/accuracy with slightly more VRAM usage.")
        parser.add_argument("--prompt_2", type=str, required=False, help="Optional different prompt for CLIP")
        parser.add_argument("--te_multiplier", nargs=2, metavar=("llm_multiplier", "clip_multiplier"), help="Scale clip and llm influence")
    elif DIFFUSION_MODEL == "framepack":
        parser.add_argument("--preview_latent_every", type=int, default=None, help="Enable latent preview every N sections. If --preview_vae is not specified it will use latent2rgb")

    if DIFFUSION_MODEL in ["wan", "hunyuan"]:
        parser.add_argument("--riflex_index", type=int, default=0, help="Frequency for RifleX extension. 4 is good for Hunyuan, 6 is good for Wan. Only 'comfy' rope_func supports this with Wan!")
        parser.add_argument("--cfgzerostar_scaling", action="store_true", help="Enables CFG-Zero* scaling - https://github.com/WeichenFan/CFG-Zero-star")
        parser.add_argument("--cfgzerostar_init_steps", type=int, default=-1, help="Enables CFGZero* zeroing out the first N steps. 2 is good for Wan T2V, 1 for I2V")
        parser.add_argument("--preview_latent_every", type=int, default=None, help="Enable latent preview every N steps. If --preview_vae is not specified it will use latent2rgb")

    # Common

    parser.add_argument("--preview_vae", type=str, help="Path to TAE vae for taehv previews")
    parser.add_argument("--cfg_schedule", type=str, help=CFG_SCHEDULE_HELP)
    parser.add_argument("--keep_pngs", action="store_true", help="Save frames as PNGs in addition to output video")
    parser.add_argument("--codec", choices=["prores", "h264", "h265"], default=None, help="Codec to use, choose from 'prores', 'h264', or 'h265'")
    parser.add_argument("--container", choices=["mkv", "mp4"], default="mkv", help="Container format to use, choose from 'mkv' or 'mp4'. Note prores can only go in MKV!")
    parser.add_argument("--fp16_accumulation", action="store_true", help="Enable full FP16 Accmumulation in FP16 GEMMs, requires Pytorch 2.7.0 or higher")
    return parser


def parse_blissful_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.seed is not None:
        try:
            args.seed = int(args.seed)
        except ValueError:
            string_seed = args.seed
            args.seed = string_to_seed(args.seed)
            logger.info(f"Seed {args.seed} was generated from string '{string_seed}'!")
    if DIFFUSION_MODEL == "wan":
        if args.riflex_index != 0 and args.rope_func.lower() != "comfy":
            logger.error("RIFLEx can only be used with rope_func == 'comfy'!")
            raise ValueError("RIFLEx can only be used with rope_func =='comfy'!")
    if DIFFUSION_MODEL in ["wan", "hunyuan"]:
        if args.cfg_schedule:
            args.cfg_schedule = parse_scheduled_cfg(args.cfg_schedule, args.infer_steps, args.guidance_scale)
        if args.cfgzerostar_scaling or args.cfgzerostar_init_steps != -1:
            if args.guidance_scale == 1 and not args.cfg_schedule:
                error_out(AttributeError, "Requested CFGZero* but CFG is not enabled so it will have no effect!")
    blissful_prefunc(args)
    return args
