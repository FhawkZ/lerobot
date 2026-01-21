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

import logging
import threading
import time
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_fr3_leader import FR3LeaderConfig

logger = logging.getLogger(__name__)


class FR3Leader(Teleoperator):
    """FR3 leader arm teleoperator using ROS2 topics."""

    config_class = FR3LeaderConfig
    name = "fr3_leader"

    def __init__(self, config: FR3LeaderConfig):
        super().__init__(config)
        self.config = config
        self._node = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._connected = False
        self._owns_rclpy = False
        self._lock = threading.Lock()
        self._joint_state_msg: Optional[JointState] = None
        self._gripper_state_msg: Optional[JointState] = None

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"{joint}.pos": float for joint in self.config.joint_names}
        if self.config.use_gripper:
            features["gripper.width"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        node_name = f"fr3_leader_{self.id or 'default'}"
        self._node = rclpy.create_node(node_name)

        self._node.create_subscription(
            JointState, self.config.joint_state_topic, self._joint_state_cb, 10
        )
        if self.config.use_gripper:
            self._node.create_subscription(
                JointState, self.config.gripper_state_topic, self._gripper_state_cb, 10
            )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._connected = True
        logger.info(
            "%s connected (joint_state=%s, gripper_state=%s)",
            self,
            self.config.joint_state_topic,
            self.config.gripper_state_topic,
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s does not require calibration", self)

    def configure(self) -> None:
        pass

    def _joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._joint_state_msg = msg

    def _gripper_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._gripper_state_msg = msg

    def _wait_for_state(self, require_gripper: bool) -> None:
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                joint_ready = self._joint_state_msg is not None
                gripper_ready = self._gripper_state_msg is not None
            if joint_ready and (not require_gripper or gripper_ready):
                return
            time.sleep(0.01)
        missing = "joint state" if not joint_ready else "gripper state"
        raise TimeoutError(f"Timeout waiting for {missing} on ROS2 topics")

    def _ordered_joint_positions(self) -> list[float]:
        with self._lock:
            msg = self._joint_state_msg
        if msg is None:
            raise RuntimeError("Joint state is not available")

        if msg.name and len(msg.name) == len(msg.position):
            name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
            return [float(name_to_pos[joint]) for joint in self.config.joint_names]

        if len(msg.position) < len(self.config.joint_names):
            raise ValueError("JointState message has fewer positions than expected")

        return [float(pos) for pos in msg.position[: len(self.config.joint_names)]]

    def _gripper_width(self) -> Optional[float]:
        with self._lock:
            msg = self._gripper_state_msg
        if msg is None or not msg.position:
            return None

        if len(msg.position) >= 2:
            width = float(msg.position[0] + msg.position[1])
        else:
            width = float(msg.position[0] * 2.0)

        return width * self.config.gripper_width_scale

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._wait_for_state(require_gripper=self.config.use_gripper)
        joint_positions = self._ordered_joint_positions()
        action = {
            f"{joint}.pos": joint_positions[idx] for idx, joint in enumerate(self.config.joint_names)
        }

        if self.config.use_gripper:
            gripper_width = self._gripper_width()
            if gripper_width is None:
                raise RuntimeError("Gripper state is not available")
            action["gripper.width"] = gripper_width

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._executor and self._node:
            self._executor.remove_node(self._node)
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

        self._connected = False
        self._executor = None
        self._node = None
        self._spin_thread = None
        logger.info("%s disconnected", self)

