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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("tavla05")
@dataclass
class Tavla05Config(PreTrainedConfig):
    """
    Configuration class for TA-VLA (Task-Aware Vision-Language-Action) policy based on PI0.5.
    
    TA-VLA extends PI0.5 with task-aware capabilities and specific data processing
    for OpenArm robots. The model expects:
    - Base camera image (cam_high)
    - Optional wrist cameras (cam_left_wrist, cam_right_wrist)
    - Robot state (14 dims, padded to action_dim)
    - Optional effort/torque signals
    - Actions (14 dims, padded to action_dim)
    
    Key differences from PI0-based TA-VLA:
    - Uses PI0.5's AdaRMS and time_mlp for time conditioning
    - Uses QUANTILES normalization (PI0.5 style)
    - Tokenizer length: 200 (PI0.5 style)
    - Adds state_proj layer (PI0.5 doesn't have it by default, but TA-VLA needs it)
    """
    
    # Model architecture (inherits from PI0.5 base)
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"
    
    # TA-VLA specific: action dimension (OpenArm uses 14 joints)
    action_dim: int = 14  # Real action dim for OpenArm
    max_action_dim: int = 32  # Padded dimension for model
    
    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict (action_horizon)
    n_action_steps: int = 50  # Number of action steps to execute
    
    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    
    # Flow matching parameters (same as PI0/PI0.5)
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    
    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )
    
    # Camera configuration for TA-VLA
    # TA-VLA expects: base_0_rgb (required), left_wrist_0_rgb, right_wrist_0_rgb (optional)
    use_wrist_cameras: bool = True  # Whether to use wrist cameras
    
    # Effort/torque configuration
    use_effort: bool = True  # Whether to include effort/torque signals
    effort_dim: int = 14  # Effort dimension (matches action_dim for OpenArm)
    effort_dim_in: int = 14  # Input effort dimension (can be larger if using history)
    effort_history: list[int] = field(default_factory=lambda: [0])  # History timestamps
    
    # Effort processing mode (matching TA-VLA's EffortType)
    # Options: "no", "state", "llm", "expert", "llm_his_c", "expert_his_c", "expert_fut", etc.
    effort_type: str = "expert"  # Default: pass effort to action expert
    
    # Normalization (PI0.5 style: uses QUANTILES)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # PI0.5 uses quantiles
            "ACTION": NormalizationMode.QUANTILES,  # PI0.5 uses quantiles
        }
    )
    
    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None
    
    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    
    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    
    # Scheduler settings
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    
    tokenizer_max_length: int = 200  # PI0.5 uses 200 tokens (vs 48 for PI0)
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")
        
        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
        
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        
        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) cannot be greater than max_action_dim ({self.max_action_dim})"
            )
    
    def validate_features(self) -> None:
        """Validate and set up input/output features for TA-VLA."""
        # Base camera (required)
        base_camera = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, *self.image_resolution),
        )
        self.input_features[f"{OBS_IMAGES}.cam_high"] = base_camera
        
        # Wrist cameras (optional)
        if self.use_wrist_cameras:
            left_wrist = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            right_wrist = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[f"{OBS_IMAGES}.cam_left_wrist"] = left_wrist
            self.input_features[f"{OBS_IMAGES}.cam_right_wrist"] = right_wrist
        
        # State feature (padded to max_state_dim)
        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature
        
        # Effort feature (optional)
        if self.use_effort:
            effort_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.effort_dim,),
            )
            self.input_features["observation.effort"] = effort_feature
        
        # Action feature (padded to max_action_dim, but output will be truncated to action_dim)
        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Model outputs max_action_dim
            )
            self.output_features[ACTION] = action_feature
    
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )
    
    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
    
    @property
    def observation_delta_indices(self) -> None:
        return None
    
    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))
    
    @property
    def reward_delta_indices(self) -> None:
        return None

