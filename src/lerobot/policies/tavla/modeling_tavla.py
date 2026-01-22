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
TA-VLA (Task-Aware Vision-Language-Action) Policy for LeRobot.

This policy extends PI0 with TA-VLA specific data processing:
- Handles OpenArm-specific camera naming (cam_high, cam_left_wrist, cam_right_wrist)
- Pads state/actions from 14 dims to model's max_action_dim
- Supports effort/torque signals
- Truncates output actions back to 14 dims

The model architecture is based on PI0, but with TA-VLA-specific preprocessing.
"""

from __future__ import annotations

import builtins
import logging
from collections import deque
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import (
    PI0Pytorch,
    PI0Policy,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.tavla.configuration_tavla import TavlaConfig
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)


class EffortType(str, Enum):
    """Effort processing modes, matching TA-VLA's EffortType."""
    NO = "no"
    STATE = "state"  # Put effort into state
    LLM = "llm"  # Pass effort to LLM (PaliGemma)
    EXPERT = "expert"  # Pass effort to action expert
    LLM_HIS_C = "llm_his_c"  # Concat history effort, pass to LLM
    EXPERT_HIS_C = "expert_his_c"  # Concat history effort, pass to expert
    EXPERT_FUT = "expert_fut"  # Predict future effort along with actions
    EXPERT_HIS_C_FUT = "expert_his_c_fut"  # Input history, output future effort
    EXPERT_HIS_C_L_FUT = "expert_his_c_l_fut"  # Input history as last token, output future


class TavlaPytorch(PI0Pytorch):
    """
    TA-VLA PyTorch model extending PI0 with effort/torque support.
    
    This class adds effort processing layers and integrates effort tokens
    into the prefix/suffix embeddings, following TA-VLA's implementation.
    """
    
    def __init__(self, config: TavlaConfig, rtc_processor=None):
        # Convert TavlaConfig to PI0Config for base initialization
        pi0_config = PI0Config(
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            dtype=config.dtype,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            num_inference_steps=config.num_inference_steps,
            time_sampling_beta_alpha=config.time_sampling_beta_alpha,
            time_sampling_beta_beta=config.time_sampling_beta_beta,
            time_sampling_scale=config.time_sampling_scale,
            time_sampling_offset=config.time_sampling_offset,
            min_period=config.min_period,
            max_period=config.max_period,
            image_resolution=config.image_resolution,
            normalization_mapping=config.normalization_mapping,
            gradient_checkpointing=config.gradient_checkpointing,
            compile_model=config.compile_model,
            compile_mode=config.compile_mode,
            device=config.device,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            optimizer_lr=config.optimizer_lr,
            optimizer_betas=config.optimizer_betas,
            optimizer_eps=config.optimizer_eps,
            optimizer_weight_decay=config.optimizer_weight_decay,
            optimizer_grad_clip_norm=config.optimizer_grad_clip_norm,
            scheduler_warmup_steps=config.scheduler_warmup_steps,
            scheduler_decay_steps=config.scheduler_decay_steps,
            scheduler_decay_lr=config.scheduler_decay_lr,
            tokenizer_max_length=config.tokenizer_max_length,
        )
        
        # Initialize base PI0Pytorch
        super().__init__(pi0_config, rtc_processor=rtc_processor)
        
        # Store TA-VLA specific config
        self.tavla_config = config
        self.effort_type = EffortType(config.effort_type.lower()) if config.use_effort else EffortType.NO
        self.effort_dim = config.effort_dim if config.use_effort else 0
        
        # Calculate effort_dim_in based on effort_type and history
        if self.effort_type in (EffortType.LLM_HIS_C, EffortType.EXPERT_HIS_C, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            history_len = len(config.effort_history) if config.effort_history else 1
            self.effort_dim_in = history_len * config.effort_dim if config.use_effort else 0
        else:
            self.effort_dim_in = config.effort_dim if config.use_effort else 0
        
        # Get action expert config for effort projection dimensions
        from lerobot.policies.pi0.modeling_pi0 import get_gemma_config
        action_expert_config = get_gemma_config(config.action_expert_variant)
        paligemma_config = get_gemma_config(config.paligemma_variant)
        expert_width = action_expert_config.width
        llm_width = paligemma_config.width
        
        # Initialize effort projection layers based on effort_type
        if self.effort_type in (EffortType.LLM, EffortType.LLM_HIS_C):
            self.effort_proj_in = nn.Linear(self.effort_dim_in, 2 * llm_width)
            self.effort_proj_out = nn.Linear(2 * llm_width, llm_width)
        elif self.effort_type in (
            EffortType.EXPERT,
            EffortType.EXPERT_HIS_C,
            EffortType.EXPERT_FUT,
            EffortType.EXPERT_HIS_C_FUT,
            EffortType.EXPERT_HIS_C_L_FUT,
        ):
            self.effort_proj_in = nn.Linear(self.effort_dim_in, 2 * expert_width)
            self.effort_proj_out = nn.Linear(2 * expert_width, expert_width)
        else:
            self.effort_proj_in = None
            self.effort_proj_out = None
        
        # If predicting future effort, modify action projections
        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            self.action_in_proj = nn.Linear(config.max_action_dim + self.effort_dim, expert_width)
            self.action_out_proj = nn.Linear(expert_width, config.max_action_dim + self.effort_dim)
        
        # Initialize effort projection weights
        if self.effort_proj_in is not None:
            nn.init.xavier_uniform_(self.effort_proj_in.weight)
            nn.init.zeros_(self.effort_proj_in.bias)
            nn.init.xavier_uniform_(self.effort_proj_out.weight)
            nn.init.zeros_(self.effort_proj_out.bias)
    
    def _project_effort(self, effort: Tensor) -> Tensor:
        """Project effort to embedding space."""
        if self.effort_proj_in is None:
            raise ValueError(f"Effort projection not initialized for effort_type={self.effort_type}")
        
        original_shape = effort.shape
        if effort.dim() == 3:  # [B, history, effort_dim]
            effort = effort.reshape(original_shape[0], -1)  # [B, history * effort_dim]
        
        effort_hidden = self.effort_proj_in(effort)
        effort_hidden = F.silu(effort_hidden)
        effort_emb = self.effort_proj_out(effort_hidden)
        
        if len(original_shape) == 3:
            emb_dim = effort_emb.shape[-1]
            effort_emb = effort_emb.reshape(original_shape[0], original_shape[1], emb_dim)
        else:
            effort_emb = effort_emb[:, None, :]
        
        return effort_emb
    
    def _process_effort_tokens(
        self, effort: Tensor | None, mode: Literal["prefix", "suffix"]
    ) -> tuple[list[Tensor], list[Tensor], list[int]]:
        """Process effort into tokens for prefix or suffix."""
        tokens_list = []
        mask_list = []
        ar_mask_list = []
        
        if effort is None or self.effort_type == EffortType.NO:
            return tokens_list, mask_list, ar_mask_list
        
        should_add = False
        if mode == "prefix" and self.effort_type in (EffortType.LLM, EffortType.LLM_HIS_C):
            should_add = True
        elif mode == "suffix" and self.effort_type in (
            EffortType.EXPERT,
            EffortType.EXPERT_HIS_C,
            EffortType.EXPERT_FUT,
            EffortType.EXPERT_HIS_C_FUT,
        ):
            should_add = True
        
        if not should_add:
            return tokens_list, mask_list, ar_mask_list
        
        ar_mask_value = 0 if mode == "prefix" else 1
        
        if self.effort_type in (EffortType.LLM, EffortType.EXPERT):
            if effort.dim() == 3:
                current_effort = effort[:, -1, :]
            else:
                current_effort = effort
            effort_token = self._project_effort(current_effort)
            tokens_list.append(effort_token)
            mask_list.append(torch.ones(effort_token.shape[:2], dtype=torch.bool, device=effort.device))
            ar_mask_list.append(ar_mask_value)
        
        elif self.effort_type in (EffortType.LLM_HIS_C, EffortType.EXPERT_HIS_C, EffortType.EXPERT_HIS_C_FUT):
            if effort.dim() == 3:
                effort_flat = effort.reshape(effort.shape[0], -1)
            else:
                effort_flat = effort
            effort_token = self._project_effort(effort_flat)
            tokens_list.append(effort_token)
            mask_list.append(torch.ones(effort_token.shape[:2], dtype=torch.bool, device=effort.device))
            ar_mask_list.append(ar_mask_value)
        
        return tokens_list, mask_list, ar_mask_list
    
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, effort: Tensor | None = None):
        """Embed prefix (images, language, effort) for LLM processing."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        effort_tokens, effort_masks, effort_ar_masks = self._process_effort_tokens(effort, mode="prefix")
        
        if effort_tokens:
            effort_emb = torch.cat(effort_tokens, dim=1)
            effort_mask = torch.cat(effort_masks, dim=1)
            prefix_embs = torch.cat([prefix_embs, effort_emb], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, effort_mask], dim=1)
            effort_ar_mask_tensor = torch.tensor(
                effort_ar_masks, dtype=torch.bool, device=prefix_att_masks.device
            )
            effort_ar_mask_tensor = effort_ar_mask_tensor[None, :].expand(prefix_embs.shape[0], len(effort_ar_masks))
            prefix_att_masks = torch.cat([prefix_att_masks, effort_ar_mask_tensor], dim=1)
        
        return prefix_embs, prefix_pad_masks, prefix_att_masks
    
    def embed_suffix(
        self, state, noisy_actions, timestep, effort: Tensor | None = None, effort_position: str = "before_state"
    ):
        """Embed suffix (state, actions, effort) for action expert processing."""
        embs = []
        pad_masks = []
        att_masks = []
        
        if effort_position == "before_state" and self.effort_type != EffortType.EXPERT_HIS_C_L_FUT:
            effort_tokens, effort_masks, effort_ar_masks = self._process_effort_tokens(effort, mode="suffix")
            if effort_tokens:
                effort_emb = torch.cat(effort_tokens, dim=1)
                effort_mask = torch.cat(effort_masks, dim=1)
                embs.append(effort_emb)
                pad_masks.append(effort_mask)
                att_masks.extend(effort_ar_masks)
        
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)
        
        def state_proj_func(state):
            return self.state_proj(state)
        
        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device
        
        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks.append(1)
        
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)
        
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)
        
        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        
        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)
        
        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        
        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
        att_masks.extend([1] + ([0] * (self.config.chunk_size - 1)))
        
        if effort_position == "after_actions" and self.effort_type == EffortType.EXPERT_HIS_C_L_FUT:
            effort_tokens, effort_masks, effort_ar_masks = self._process_effort_tokens(effort, mode="suffix")
            if effort_tokens:
                effort_emb = torch.cat(effort_tokens, dim=1)
                effort_mask = torch.cat(effort_masks, dim=1)
                embs.append(effort_emb)
                pad_masks.append(effort_mask)
                att_masks.extend(effort_ar_masks)
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        adarms_cond = None
        return embs, pad_masks, att_masks, adarms_cond
    
    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        effort: Tensor | None = None,
        noise=None,
        time=None,
    ) -> Tensor:
        """Forward pass with effort support."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            future_steps = actions.shape[1]
            if actions.shape[-1] == self.tavla_config.action_dim + self.effort_dim:
                future_effort = actions[:, :, self.tavla_config.action_dim:]
                actions = actions[:, :, : self.tavla_config.action_dim]
            else:
                future_effort = None
        else:
            future_effort = None
        
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, effort=effort
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time, effort=effort
        )
        
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out
        
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        
        if self.effort_type == EffortType.EXPERT_HIS_C_L_FUT:
            suffix_out = suffix_out[:, -self.config.chunk_size - 1 : -1]
        else:
            suffix_out = suffix_out[:, -self.config.chunk_size :]
        
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)
        
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        
        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            if future_effort is not None:
                action_loss = F.mse_loss(
                    v_t[..., : self.tavla_config.action_dim],
                    u_t[..., : self.tavla_config.action_dim],
                    reduction="none",
                )
                effort_loss = F.mse_loss(
                    v_t[..., self.tavla_config.action_dim :],
                    future_effort,
                    reduction="none",
                )
                return action_loss + 0.1 * effort_loss
            else:
                return F.mse_loss(
                    v_t[..., : self.tavla_config.action_dim],
                    u_t[..., : self.tavla_config.action_dim],
                    reduction="none",
                )
        else:
            return F.mse_loss(v_t, u_t, reduction="none")
    
    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        effort: Tensor | None = None,
        noise=None,
        num_steps=None,
        **kwargs,
    ) -> Tensor:
        """Sample actions with effort support."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        
        bsize = state.shape[0]
        device = state.device
        
        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim + (self.effort_dim if self.effort_type in (
                    EffortType.EXPERT_FUT,
                    EffortType.EXPERT_HIS_C_FUT,
                    EffortType.EXPERT_HIS_C_L_FUT,
                ) else 0),
            )
            noise = self.sample_noise(actions_shape, device)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, effort=effort
        )
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        
        dt = -1.0 / num_steps
        x_t = noise
        
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            
            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    state=state,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    effort=effort,
                )
            
            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)
            
            x_t = x_t + dt * v_t
            
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        
        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            x_0 = x_t[..., : self.tavla_config.action_dim]
        else:
            x_0 = x_t
        
        return x_0
    
    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        effort: Tensor | None = None,
    ):
        """Apply one denoising step with effort support."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep, effort=effort
        )
        
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        
        suffix_out = outputs_embeds[1]
        
        if self.effort_type == EffortType.EXPERT_HIS_C_L_FUT:
            suffix_out = suffix_out[:, -self.config.chunk_size - 1 : -1]
        else:
            suffix_out = suffix_out[:, -self.config.chunk_size :]
        
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class TavlaPolicy(PreTrainedPolicy):
    """
    TA-VLA Policy for LeRobot.
    
    This policy wraps PI0 with TA-VLA-specific data processing:
    - Maps OpenArm camera names to TA-VLA format
    - Handles state/action padding/truncation
    - Supports effort/torque signals
    """
    
    config_class = TavlaConfig
    name = "tavla"
    
    def __init__(
        self,
        config: TavlaConfig,
        **kwargs,
    ):
        """
        Args:
            config: TA-VLA configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        
        # Convert TavlaConfig to PI0Config for the underlying model
        # TA-VLA uses PI0 architecture but with different data processing
        pi0_config = PI0Config(
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            dtype=config.dtype,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            num_inference_steps=config.num_inference_steps,
            time_sampling_beta_alpha=config.time_sampling_beta_alpha,
            time_sampling_beta_beta=config.time_sampling_beta_beta,
            time_sampling_scale=config.time_sampling_scale,
            time_sampling_offset=config.time_sampling_offset,
            min_period=config.min_period,
            max_period=config.max_period,
            image_resolution=config.image_resolution,
            normalization_mapping=config.normalization_mapping,
            gradient_checkpointing=config.gradient_checkpointing,
            compile_model=config.compile_model,
            compile_mode=config.compile_mode,
            device=config.device,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            optimizer_lr=config.optimizer_lr,
            optimizer_betas=config.optimizer_betas,
            optimizer_eps=config.optimizer_eps,
            optimizer_weight_decay=config.optimizer_weight_decay,
            optimizer_grad_clip_norm=config.optimizer_grad_clip_norm,
            scheduler_warmup_steps=config.scheduler_warmup_steps,
            scheduler_decay_steps=config.scheduler_decay_steps,
            scheduler_decay_lr=config.scheduler_decay_lr,
            tokenizer_max_length=config.tokenizer_max_length,
        )
        pi0_config.validate_features()
        
        # Initialize the underlying TA-VLA model with effort support
        # Initialize RTC processor if configured (same as PI0Policy)
        self.rtc_processor = None
        if pi0_config.rtc_config is not None:
            from lerobot.policies.rtc.modeling_rtc import RTCProcessor
            self.rtc_processor = RTCProcessor(pi0_config.rtc_config)
        
        # Use TavlaPytorch instead of PI0Pytorch for effort support
        self.model = TavlaPytorch(config, rtc_processor=self.rtc_processor)
        
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.to(config.device if config.device else "cpu")
        
        # Initialize observation queues for temporal processing
        self.reset()
    
    def reset(self) -> None:
        """Reset the policy's internal state (observation queues)."""
        # Initialize queues for temporal observations
        # TA-VLA uses n_obs_steps=1, so we only need single-frame queues
        self.obs_image_queues: dict[str, deque] = {}
        self.obs_state_queue: deque = deque(maxlen=self.config.n_obs_steps)
        self.obs_effort_queue: deque | None = None
        if self.config.use_effort:
            self.obs_effort_queue = deque(maxlen=self.config.n_obs_steps)
    
    def select_action(
        self,
        batch: dict[str, Tensor],
        **kwargs,
    ) -> Tensor:
        """
        Select action from observation batch.
        
        Args:
            batch: Observation batch in TA-VLA format (after preprocessing):
                - image: dict with base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
                - image_mask: dict with boolean masks
                - state: [B, max_state_dim]
                - effort: [B, effort_dim] or [B, history, effort_dim] (optional)
            **kwargs: Additional arguments passed to model
            
        Returns:
            Action tensor of shape [B, chunk_size, max_action_dim]
            (will be truncated to action_dim in post-processing)
        """
        # The batch is already in TA-VLA format from preprocessor
        # Convert to model format for TavlaPytorch
        model_batch = self._convert_tavla_to_model_batch(batch)
        
        # Extract components
        images = model_batch.get("images", [])
        img_masks = model_batch.get("img_masks", [])
        lang_tokens = model_batch.get("lang_tokens", None)
        lang_masks = model_batch.get("lang_masks", None)
        state = model_batch["state"]
        effort = model_batch.get("effort", None)
        
        # Call TavlaPytorch model with effort support
        actions = self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens if lang_tokens is not None else torch.zeros((state.shape[0], 0), dtype=torch.long, device=state.device),
            lang_masks=lang_masks if lang_masks is not None else torch.zeros((state.shape[0], 0), dtype=torch.bool, device=state.device),
            state=state,
            effort=effort,
            **kwargs,
        )
        
        return actions
    
    def _convert_tavla_to_model_batch(self, batch: dict[str, Tensor | dict]) -> dict[str, Tensor | dict]:
        """
        Convert TA-VLA format batch to model format (TavlaPytorch expects specific format).
        
        TA-VLA format:
            - image: dict with base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
            - image_mask: dict with boolean masks
            - state: [B, max_state_dim]
            - effort: [B, effort_dim] or [B, history, effort_dim] (optional)
            
        Model format (for TavlaPytorch):
            - images: list of [B, H, W, C] tensors
            - img_masks: list of [B] boolean tensors
            - lang_tokens: [B, seq_len] (optional)
            - lang_masks: [B, seq_len] (optional)
            - state: [B, max_state_dim]
            - effort: [B, effort_dim] or [B, history, effort_dim] (optional)
        """
        model_batch = {}
        
        # Convert images to list format
        if "image" in batch:
            images = batch["image"]
            image_list = []
            mask_list = []
            # Order: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
            for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
                if key in images:
                    img = images[key]
                    # Ensure [B, H, W, C] format
                    if img.dim() == 4 and img.shape[1] == 3:  # [B, C, H, W]
                        img = img.permute(0, 2, 3, 1)  # [B, H, W, C]
                    image_list.append(img)
                    # Get corresponding mask
                    if "image_mask" in batch and key in batch["image_mask"]:
                        mask_list.append(batch["image_mask"][key])
                    else:
                        # Create default mask (all True)
                        mask_list.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))
            
            model_batch["images"] = image_list
            model_batch["img_masks"] = mask_list
        
        # Language tokens (if provided)
        if "tokenized_prompt" in batch:
            model_batch["lang_tokens"] = batch["tokenized_prompt"]
            model_batch["lang_masks"] = batch.get("tokenized_prompt_mask", None)
        else:
            # Create empty language tokens if not provided
            if "state" in batch:
                bsize = batch["state"].shape[0]
                device = batch["state"].device
                model_batch["lang_tokens"] = torch.zeros((bsize, 0), dtype=torch.long, device=device)
                model_batch["lang_masks"] = torch.zeros((bsize, 0), dtype=torch.bool, device=device)
        
        # Convert state
        if "state" in batch:
            model_batch["state"] = batch["state"]
        
        # Convert effort (keep as-is, TavlaPytorch will handle it)
        if "effort" in batch and self.config.use_effort:
            model_batch["effort"] = batch["effort"]
        
        return model_batch
    
    def compute_loss(
        self,
        batch: dict[str, Tensor],
        **kwargs,
    ) -> dict[str, Tensor]:
        """
        Compute training loss.
        
        Args:
            batch: Training batch in TA-VLA format (after preprocessing)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with loss values
        """
        # Convert to model format
        model_batch = self._convert_tavla_to_model_batch(batch)
        
        # Extract components
        images = model_batch.get("images", [])
        img_masks = model_batch.get("img_masks", [])
        lang_tokens = model_batch.get("lang_tokens", None)
        lang_masks = model_batch.get("lang_masks", None)
        state = model_batch["state"]
        effort = model_batch.get("effort", None)
        actions = batch.get("actions", None)
        
        if actions is None:
            raise ValueError("Actions are required for computing loss")
        
        # Actions should be [B, chunk_size, max_action_dim]
        # Pad if needed
        if actions.shape[-1] < self.config.max_action_dim:
            actions = torch.nn.functional.pad(
                actions, (0, self.config.max_action_dim - actions.shape[-1]), mode="constant", value=0
            )
        
        # Call TavlaPytorch forward with effort support
        loss = self.model.forward(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens if lang_tokens is not None else torch.zeros((state.shape[0], 0), dtype=torch.long, device=state.device),
            lang_masks=lang_masks if lang_masks is not None else torch.zeros((state.shape[0], 0), dtype=torch.bool, device=state.device),
            state=state,
            actions=actions,
            effort=effort,
            **kwargs,
        )
        
        return {"loss": loss}
    
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """
        Load a pretrained TA-VLA model.
        
        Note: TA-VLA models are compatible with PI0 checkpoints,
        but may require weight remapping if the architecture differs.
        """
        logger.info(
            "TA-VLA model is based on PI0 architecture. "
            "Loading pretrained weights from PI0 checkpoint..."
        )
        
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")
        
        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        
        # Initialize model
        model = cls(config, **kwargs)
        
        # Try to load PI0 weights (TA-VLA is compatible with PI0)
        try:
            # Use PI0's from_pretrained logic for weight loading
            pi0_model = PI0Policy.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                config=None,  # Will be created from PI0Config
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=strict,
                **kwargs,
            )
            
            # Copy weights from PI0 model to TA-VLA model
            model.model.load_state_dict(pi0_model.model.state_dict(), strict=strict)
            logger.info("Successfully loaded pretrained weights from PI0 checkpoint")
            
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
            logger.info("Returning model with random initialization")
        
        return model

