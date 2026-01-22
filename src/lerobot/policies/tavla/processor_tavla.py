#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TA-VLA specific data processors.

This module handles the data transformation between LeRobot format and TA-VLA format:
- Maps camera names: cam_high -> base_0_rgb, cam_left_wrist -> left_wrist_0_rgb, etc.
- Pads state and actions from action_dim (14) to max_action_dim (32)
- Handles effort/torque signals
- Truncates output actions from max_action_dim back to action_dim
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from lerobot.policies.tavla.configuration_tavla import TavlaConfig
from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)


def pad_to_dim(vector: torch.Tensor | np.ndarray, target_dim: int) -> torch.Tensor | np.ndarray:
    """Pad the last dimension of a vector to target_dim with zeros.
    
    Args:
        vector: Input vector of shape [..., current_dim]
        target_dim: Target dimension to pad to
        
    Returns:
        Padded vector of shape [..., target_dim]
    """
    if isinstance(vector, np.ndarray):
        current_dim = vector.shape[-1]
        if current_dim >= target_dim:
            return vector
        pad_width = [(0, 0)] * (vector.ndim - 1) + [(0, target_dim - current_dim)]
        return np.pad(vector, pad_width, mode="constant", constant_values=0)
    else:  # torch.Tensor
        current_dim = vector.shape[-1]
        if current_dim >= target_dim:
            return vector
        return torch.nn.functional.pad(vector, (0, target_dim - current_dim), mode="constant", value=0)


def truncate_to_dim(vector: torch.Tensor | np.ndarray, target_dim: int) -> torch.Tensor | np.ndarray:
    """Truncate the last dimension of a vector to target_dim.
    
    Args:
        vector: Input vector of shape [..., current_dim]
        target_dim: Target dimension to truncate to
        
    Returns:
        Truncated vector of shape [..., target_dim]
    """
    return vector[..., :target_dim]


class TavlaInputProcessor:
    """Processes inputs for TA-VLA model.
    
    Converts LeRobot observation format to TA-VLA format:
    - Maps camera names
    - Pads state to max_action_dim
    - Handles effort signals
    """
    
    def __init__(self, config: TavlaConfig):
        self.config = config
        self.action_dim = config.action_dim
        self.max_action_dim = config.max_action_dim
        self.max_state_dim = config.max_state_dim
        self.use_effort = config.use_effort
        self.use_wrist_cameras = config.use_wrist_cameras
        
        # Camera name mapping: LeRobot -> TA-VLA
        self.camera_mapping = {
            "cam_high": "base_0_rgb",
        }
        if self.use_wrist_cameras:
            self.camera_mapping.update({
                "cam_left_wrist": "left_wrist_0_rgb",
                "cam_right_wrist": "right_wrist_0_rgb",
            })
    
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process input batch for TA-VLA model.
        
        Args:
            batch: LeRobot format batch with keys like:
                - observation.images.cam_high: [B, 3, H, W]
                - observation.images.cam_left_wrist: [B, 3, H, W] (optional)
                - observation.images.cam_right_wrist: [B, 3, H, W] (optional)
                - observation.state: [B, state_dim]
                - observation.effort: [B, effort_dim] (optional)
                
        Returns:
            TA-VLA format batch with keys:
                - image: dict with base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
                - image_mask: dict with boolean masks
                - state: [B, max_state_dim] (padded)
                - effort: [B, effort_dim] (if use_effort=True)
        """
        processed = {}
        
        # Process images: map LeRobot camera names to TA-VLA names
        images = {}
        image_masks = {}
        
        for lerobot_name, tavla_name in self.camera_mapping.items():
            image_key = f"observation.images.{lerobot_name}"
            if image_key in batch:
                # Convert from [B, C, H, W] to [B, H, W, C] if needed
                img = batch[image_key]
                if isinstance(img, torch.Tensor):
                    if img.dim() == 4 and img.shape[1] == 3:  # [B, C, H, W]
                        img = img.permute(0, 2, 3, 1)  # [B, H, W, C]
                    # Normalize to [-1, 1] if in [0, 255] or [0, 1]
                    if img.dtype == torch.uint8:
                        img = img.float() / 255.0 * 2.0 - 1.0
                    elif img.max() <= 1.0 and img.min() >= 0.0:
                        img = img * 2.0 - 1.0
                elif isinstance(img, np.ndarray):
                    if img.ndim == 4 and img.shape[1] == 3:  # [B, C, H, W]
                        img = np.transpose(img, (0, 2, 3, 1))  # [B, H, W, C]
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                    elif img.max() <= 1.0 and img.min() >= 0.0:
                        img = img.astype(np.float32) * 2.0 - 1.0
                
                images[tavla_name] = img
                # Create mask: True if image is valid (not all zeros/black)
                if isinstance(img, torch.Tensor):
                    mask = torch.any(img != 0, dim=(1, 2, 3))
                else:
                    mask = np.any(img != 0, axis=(1, 2, 3))
                image_masks[tavla_name] = mask
            else:
                # Create empty image if not provided
                if isinstance(batch.get("observation.images.cam_high"), torch.Tensor):
                    img_shape = batch["observation.images.cam_high"].shape
                    if img_shape[1] == 3:  # [B, C, H, W]
                        empty_img = torch.zeros(
                            (img_shape[0], img_shape[2], img_shape[3], img_shape[1]),
                            dtype=torch.float32,
                            device=batch["observation.images.cam_high"].device
                        )
                    else:  # [B, H, W, C]
                        empty_img = torch.zeros(
                            img_shape,
                            dtype=torch.float32,
                            device=batch["observation.images.cam_high"].device
                        )
                    images[tavla_name] = empty_img
                    image_masks[tavla_name] = torch.zeros(img_shape[0], dtype=torch.bool, device=empty_img.device)
                else:
                    img_shape = batch["observation.images.cam_high"].shape
                    if img_shape[1] == 3:  # [B, C, H, W]
                        empty_img = np.zeros(
                            (img_shape[0], img_shape[2], img_shape[3], img_shape[1]),
                            dtype=np.float32
                        )
                    else:  # [B, H, W, C]
                        empty_img = np.zeros(img_shape, dtype=np.float32)
                    images[tavla_name] = empty_img
                    image_masks[tavla_name] = np.zeros(img_shape[0], dtype=bool)
        
        processed["image"] = images
        processed["image_mask"] = image_masks
        
        # Process state: pad to max_state_dim
        if "observation.state" in batch:
            state = batch["observation.state"]
            processed["state"] = pad_to_dim(state, self.max_state_dim)
        else:
            # Create zero state if not provided
            batch_size = next(iter(images.values())).shape[0]
            if isinstance(next(iter(images.values())), torch.Tensor):
                processed["state"] = torch.zeros(
                    (batch_size, self.max_state_dim),
                    dtype=torch.float32,
                    device=next(iter(images.values())).device
                )
            else:
                processed["state"] = np.zeros((batch_size, self.max_state_dim), dtype=np.float32)
        
        # Process effort: include if use_effort=True
        if self.use_effort and "observation.effort" in batch:
            processed["effort"] = batch["observation.effort"]
        
        return processed


class TavlaOutputProcessor:
    """Processes outputs from TA-VLA model.
    
    Converts TA-VLA action format back to LeRobot format:
    - Truncates actions from max_action_dim back to action_dim
    """
    
    def __init__(self, config: TavlaConfig):
        self.config = config
        self.action_dim = config.action_dim
        self.max_action_dim = config.max_action_dim
    
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process output batch from TA-VLA model.
        
        Args:
            batch: TA-VLA format batch with:
                - actions: [B, chunk_size, max_action_dim]
                
        Returns:
            LeRobot format batch with:
                - action: [B, chunk_size, action_dim] (truncated)
        """
        processed = {}
        
        if "actions" in batch:
            # Truncate from max_action_dim to action_dim
            actions = batch["actions"]
            processed["action"] = truncate_to_dim(actions, self.action_dim)
        else:
            logger.warning("No 'actions' key found in batch, returning empty dict")
        
        return processed


def make_tavla_pre_post_processors(
    config: TavlaConfig,
    **kwargs,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Create pre- and post-processors for TA-VLA policy.
    
    Args:
        config: TA-VLA configuration
        **kwargs: Additional arguments (unused for now)
        
    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    preprocessor = PolicyProcessorPipeline([TavlaInputProcessor(config)])
    postprocessor = PolicyProcessorPipeline([TavlaOutputProcessor(config)])
    
    return preprocessor, postprocessor

