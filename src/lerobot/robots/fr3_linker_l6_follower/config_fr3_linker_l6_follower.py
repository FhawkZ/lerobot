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

    # Use FollowJointTrajectory style arm command, consistent with fr3_follower.
    joint_command_topic: str = "/NS_1/fr3_arm_controller/joint_trajectory"
    # Keep arm feedback aligned with fr3_follower measured states.
    joint_state_topic: str = "/NS_1/franka_robot_state_broadcaster/measured_joint_states"
    hand_control_topic: str = "/cb_right_hand_control_cmd"
    hand_state_topic: str = "/cb_right_hand_state"
    enable_arm_publish: bool = True
    # Match lerobot `dataset.fps` / teleop loop. Each trajectory point uses
    # time_from_start = 1/control_hz so the segment length matches one command cycle.
    control_hz: float = 60.0
    # EMA smoothing factor for arm commands: lower is smoother, higher is more responsive.
    ema_alpha: float = 0.15
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
