#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("mocap_leader")
@dataclass
class MocapLeaderConfig(TeleoperatorConfig):
    """Configuration for MocapLeader (PNstudio motion capture)."""

    joint_state_topic: str = "/joint_states"
    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]
    )
    hand_joint_names: list[str] = field(
        default_factory=lambda: [
            "hand_0",
            "hand_1",
            "hand_2",
            "hand_3",
            "hand_4",
            "hand_5",
        ]
    )
    timeout_s: float = 5.0
