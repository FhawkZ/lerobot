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


@TeleoperatorConfig.register_subclass("fr3_leader")
@dataclass
class FR3LeaderConfig(TeleoperatorConfig):
    """Configuration for Franka Research 3 leader arm (ROS2)."""

    joint_state_topic: str = "/NS_2/franka_robot_state_broadcaster/measured_joint_states"
    gripper_state_topic: str = "/NS_2/franka_gripper/joint_states"
    joint_names: list[str] = field(
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
    use_gripper: bool = True
    gripper_width_scale: float = 1.0
    timeout_s: float = 5.0

