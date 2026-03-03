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
from .config_mocap_leader import MocapLeaderConfig

logger = logging.getLogger(__name__)


class MocapLeader(Teleoperator):
    """Mocap leader teleoperator subscribing to /joint_states from mocap_ros_py (PNstudio)."""

    config_class = MocapLeaderConfig
    name = "mocap_leader"

    def __init__(self, config: MocapLeaderConfig):
        super().__init__(config)
        self.config = config
        self._node = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._connected = False
        self._owns_rclpy = False
        self._lock = threading.Lock()
        self._joint_state_msg: Optional[JointState] = None

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"{j}.pos": float for j in self.config.arm_joint_names}
        features.update({f"{j}.pos": float for j in self.config.hand_joint_names})
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

        node_name = f"mocap_leader_{self.id or 'default'}"
        self._node = rclpy.create_node(node_name)
        self._node.create_subscription(
            JointState, self.config.joint_state_topic, self._joint_state_cb, 10
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._connected = True
        logger.info("%s connected (joint_state=%s)", self, self.config.joint_state_topic)

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

    def _wait_for_state(self) -> None:
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._joint_state_msg is not None:
                    return
            time.sleep(0.01)
        raise TimeoutError("Timeout waiting for joint state on ROS2 topic")

    def _extract_positions(self, names: list[str], start_index: int = 0) -> list[float]:
        with self._lock:
            msg = self._joint_state_msg
        if msg is None:
            raise RuntimeError("Joint state is not available")

        if msg.name and len(msg.name) == len(msg.position):
            name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
            return [float(name_to_pos.get(j, 0.0)) for j in names]

        end = start_index + len(names)
        if len(msg.position) < end:
            raise ValueError("JointState has fewer positions than expected")
        return [float(msg.position[i]) for i in range(start_index, end)]

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self._wait_for_state()
        arm_pos = self._extract_positions(self.config.arm_joint_names, 0)
        hand_pos = self._extract_positions(
            self.config.hand_joint_names, len(self.config.arm_joint_names)
        )

        action = {f"{j}.pos": arm_pos[i] for i, j in enumerate(self.config.arm_joint_names)}
        action.update({f"{j}.pos": hand_pos[i] for i, j in enumerate(self.config.hand_joint_names)})
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

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
