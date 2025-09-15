#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Blissful Tuner extension
License: Apache 2.0
Created on Sat Apr 12 14:09:37 2025

@author: blyss
"""
import argparse
import hashlib
import torch
import safetensors
from typing import List, Union, Dict, Tuple, Optional
import logging
from rich.logging import RichHandler


# Adapted from ComfyUI
def load_torch_file(
    ckpt: str,
    safe_load: Optional[bool] = True,
    device: Optional[Union[str, torch.device]] = None,
    return_metadata: Optional[bool] = False
) -> Union[
    Dict[str, torch.Tensor],
    Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]
]:
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, ckpt))
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, ckpt))
            raise e
    else:

        pl_sd = torch.load(ckpt, map_location=device, weights_only=safe_load)

        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd


def add_noise_to_reference_video(
    image: torch.Tensor,
    ratio: Optional[float] = None
) -> torch.Tensor:
    """
    Add Gaussian noise (scaled by `ratio`) to an image or batch of images.
    Supports:
      • Single image:   (C, H, W)
      • Batch of images: (B, C, H, W)
    Any pixel exactly == –1 will have zero noise (mask value).
    """
    if ratio is None or ratio == 0.0:
        return image

    dims = image.ndim
    if dims == 3:
        # Single image -> make it a batch of 1
        image = image.unsqueeze(0)  # -> (1, C, H, W)
        squeeze_back = True
    elif dims == 4:
        squeeze_back = False
    else:
        raise ValueError(
            f"add_noise_to_reference_video() expected 3D or 4D tensor, got {dims}D"
        )

    # image is now (B, C, H, W)
    B, C, H, W = image.shape
    # make a (B,) sigma array, all = ratio
    sigma = image.new_ones((B,)) * ratio
    # sample noise and scale by sigma
    noise = torch.randn_like(image) * sigma.view(B, 1, 1, 1)
    # zero out noise wherever the original was -1
    noise = torch.where(image == -1, torch.zeros_like(image), noise)

    out = image + noise
    return out.squeeze(0) if squeeze_back else out


# Below here, Blyss wrote it!
class BlissfulLogger:
    def __init__(self, logging_source: str, log_color: str, do_announce: Optional[bool] = False):
        logging_source = f"{logging_source}"
        self.logging_source = "{:<8}".format(logging_source)
        self.log_color = log_color
        self.logger = logging.getLogger(self.logging_source)
        self.logger.setLevel(logging.DEBUG)

        self.handler = RichHandler(
            show_time=False,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
            markup=True
        )

        formatter = logging.Formatter(
            f"[{self.log_color} bold]%(name)s[/] | %(message)s [dim](%(funcName)s)[/]"
        )

        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        if do_announce:
            self.logger.info("Set up logging!")

    def set_color(self, new_color):
        self.log_color = new_color
        formatter = logging.Formatter(
            f"[{self.log_color} bold]%(name)s[/] | %(message)s [dim](%(funcName)s)[/]"
        )
        self.handler.setFormatter(formatter)

    def set_name(self, new_name):
        self.logging_source = "{:<8}".format(new_name)
        self.logger = logging.getLogger(self.logging_source)
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers (just in case)
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.handler)
        else:
            self.logger.handlers.clear()
            self.logger.addHandler(self.handler)

    def info(self, msg):
        self.logger.info(msg, stacklevel=2)

    def debug(self, msg):
        self.logger.debug(msg, stacklevel=2)

    def warning(self, msg, levelmod=0):
        self.logger.warning(msg, stacklevel=2 + levelmod)

    def warn(self, msg):
        self.logger.warning(msg, stacklevel=2)

    def error(self, msg):
        self.logger.error(msg, stacklevel=2)

    def critical(self, msg):
        self.logger.critical(msg, stacklevel=2)

    def setLevel(self, level):
        self.logger.set_level(level)


def parse_scheduled_cfg(schedule: str, infer_steps: int, guidance_scale: int) -> List[int]:
    """
    Parse a schedule string like "1-10,20,!5,e~3" into a sorted list of steps.

    - "start-end" includes all steps in [start, end]
    - "e~n"    includes every nth step (n, 2n, ...) up to infer_steps
    - "x"      includes the single step x
    - Prefix "!" on any token to exclude those steps instead of including them.
    - Postfix ":float" e.g. ":6.0" to any step or range to specify a guidance_scale override for that step

    Raises argparse.ArgumentTypeError on malformed tokens or out-of-range steps.
    """
    excluded = set()
    guidance_scale_dict = {}

    for raw in schedule.split(","):
        token = raw.strip()
        if not token:
            continue  # skip empty tokens

        # exclusion if it starts with "!"
        if token.startswith("!"):
            target = "exclude"
            token = token[1:]
        else:
            target = "include"

        weight = guidance_scale
        if ":" in token:
            token, float_part = token.rsplit(":", 1)
            weight = float(float_part)

        # modulus syntax: e.g. "e~3"
        if token.startswith("e~"):
            num_str = token[2:]
            try:
                n = int(num_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid modulus in '{raw}'")
            if n < 1:
                raise argparse.ArgumentTypeError(f"Modulus must be ≥ 1 in '{raw}'")

            steps = range(n, infer_steps + 1, n)

        # range syntax: e.g. "5-10"
        elif "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"Malformed range '{raw}'")
            start_str, end_str = parts
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Non‑integer in range '{raw}'")
            if start < 1 or end < 1:
                raise argparse.ArgumentTypeError(f"Steps must be ≥ 1 in '{raw}'")
            if start > end:
                raise argparse.ArgumentTypeError(f"Start > end in '{raw}'")
            if end > infer_steps:
                raise argparse.ArgumentTypeError(f"End > infer_steps ({infer_steps}) in '{raw}'")

            steps = range(start, end + 1)

        # single‑step syntax: e.g. "7"
        else:
            try:
                step = int(token)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid token '{raw}'")
            if step < 1 or step > infer_steps:
                raise argparse.ArgumentTypeError(f"Step {step} out of range 1–{infer_steps} in '{raw}'")

            steps = [step]

        # apply include/exclude
        if target == "include":
            for step in steps:
                guidance_scale_dict[step] = weight
        else:
            excluded.update(steps)

    for step in excluded:
        guidance_scale_dict.pop(step, None)
    return guidance_scale_dict


def setup_compute_context(device: Optional[Union[torch.device, str]] = None, dtype: Optional[Union[torch.dtype, str]] = None) -> Tuple[torch.device, torch.dtype]:
    dtype_mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp8": torch.float8_e4m3fn,
        "float8": torch.float8_e4m3fn
    }
    if device is None:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.mps.is_available():
            device = torch.device("mps")
    elif isinstance(device, str):
        device = torch.device(device)

    if dtype is None:
        dtype = torch.float32
    elif isinstance(dtype, str):
        if dtype not in dtype_mapping:
            raise ValueError(f"Unknown dtype string '{dtype}'")
        dtype = dtype_mapping[dtype]

    torch.set_float32_matmul_precision('high')
    if dtype == torch.float16 or dtype == torch.bfloat16:
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            print("FP16 accumulation enabled.")
    return device, dtype


def string_to_seed(s: str, bits: int = 63) -> int:
    """
    Turn any string into a reproducible integer in [0, 2**bits) with a hash and some other logic.

    Args:
        s:           Input string
        bits:        Number of bits for the final seed (PyTorch accepts up to 63 safely, numpy likes 32)
    Returns:
        A non-negative int < 2**bits
    """
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    crypto = int.from_bytes(digest, byteorder="big")
    mask = (1 << bits) - 1
    algo = 0
    for i, char in enumerate(s):
        char_val = ord(char)
        if i % 2 == 0:
            algo *= char_val
        elif i % 3 == 0:
            algo -= char_val
        elif i % 5 == 0:
            algo /= char_val
        else:
            algo += char_val
    seed = (abs(crypto - int(algo))) & mask
    return seed


def error_out(error, message):
    logger = BlissfulLogger(__name__, "#8e00ed")
    logger.warning(message, levelmod=1)
    raise error(message)
