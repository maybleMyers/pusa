#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:51:05 2025
Prompt weighting for WanVideo
Adapted and heavily modified from https://github.com/xhinker/sd_embed
License: Apache 2.0
@author: blyss
"""
from transformers import T5Model
import torch
import re
from typing import Tuple, List, Union
from blissful_tuner.utils import BlissfulLogger
logger = BlissfulLogger(__name__, "#8e00ed")


class MiniT5Wrapper():
    """A mini wrapper for the T5 to make managing prompt weighting in Musubi easier"""

    def __init__(self, device: torch.device, dtype: torch.dtype, t5: T5Model):
        self.device = device
        self.dtype = dtype
        self.t5 = t5
        self.model = t5.model
        self.times_called = 0

    def __call__(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        max_len: int = None
    ) -> List[torch.Tensor]:
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("MiniT5Wrapper expects a single prompt at a time (wrapped as a list). Got multiple prompts.")
            prompt = prompt[0]
        if self.times_called == 0:  # Only print this notice once even if called multiple times
            logger.info("Weighting prompts...")
        # Split positive prompts and process each with weights
        prompts_raw = [p.strip() for p in prompt.split('|')]
        prompts = []
        all_weights = []

        for p in prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            prompts.append(cleaned_prompt)
            all_weights.append(weights)
        context = self.t5(prompts, device)

        # Apply weights to embeddings if any were extracted
        for i, weights in enumerate(all_weights):
            for text, weight in weights.items():
                logger.info(f"Applying weight ({weight}) to promptchunk: '{text}'")
                if len(weights) > 0:
                    context[i] = context[i] * weight
        self.times_called += 1
        return context

    def parse_prompt_weights(self, prompt: str) -> Tuple[str, dict]:
        """Extract text and weights from prompts with (text:weight) format"""
        # Parse all instances of (text:weight) in the prompt
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)

        # Replace each match with just the text part
        cleaned_prompt = prompt
        weights = {}

        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)

        return cleaned_prompt, weights
