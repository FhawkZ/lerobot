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

from ..config import RobotConfig


@RobotConfig.register_subclass("fr3_linker_l6_follower")
@dataclass
class FR3LinkerL6FollowerConfig(RobotConfig):
    """Configuration for FR3 arm + Linker L6 hand follower (ROS2)."""

    joint_command_topic: str = "/NS_1/fr3_arm_controller/joint_trajectory"
    joint_state_topic: str = "/NS_1/franka_robot_state_broadcaster/measured_joint_states"
    hand_control_topic: str = "/cb_right_hand_control_cmd"
    hand_state_topic: str = "/cb_right_hand_state"
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
    # Linker L6 uses 0-255 command range. Scale/offset to convert from action space (e.g. rad or 0-1) to 0-255.
    hand_position_scale: float = 40.5845  # 255 / (2 * pi) for rad -> [0,255]
    hand_position_offset: float = 127.5
    timeout_s: float = 5.0
