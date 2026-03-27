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

    # Optional: legacy JointState topic from mocap_to_robot_ros2 (not used by new IK path)
    joint_state_topic: str = "/joint_states"

    # FR3 arm joint names (must match FR3LinkerL6Follower.arm_joint_names)
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

    # Linker L6 hand joint names (must match FR3LinkerL6Follower.hand_joint_names order)
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

    # Path to FR3 URDF used for IK (if empty, defaults to local fr3.urdf next to mocap_leader.py)
    fr3_urdf_path: str = ""
    # Name of the FR3 end-effector link in the URDF
    fr3_ee_frame_name: str = "fr3_link8"

    # FR3 joint state topic for FK (current EE pose). Used for incremental control.
    # Keep aligned with follower default to avoid dependence on
    # franka_robot_state_broadcaster realtime publisher stability.
    fr3_joint_state_topic: str = "/NS_1/joint_states"

    # Optional axis alignment from mocap frame to FR3 frame.
    # When enabled, apply:
    #   X_robot = Y_mocap, Y_robot = -X_mocap, Z_robot = Z_mocap
    # to both translation delta and rotation-vector delta before IK.
    enable_mocap_to_fr3_axis_mapping: bool = False

    # Timeout waiting for mocap / joint state messages
    timeout_s: float = 5.0

    # Mocap background poll frequency (Hz), same as mocap_to_linkerhand MocapReader
    mocap_poll_hz: float = 120.0
