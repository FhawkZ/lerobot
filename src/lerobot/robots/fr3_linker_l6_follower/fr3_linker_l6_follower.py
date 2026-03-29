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
import math
import threading
import time
from typing import Any, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_fr3_linker_l6_follower import FR3LinkerL6FollowerConfig

logger = logging.getLogger(__name__)


def _ordered_values(msg: JointState, names: list[str], attr: str) -> list[float]:
    vals = getattr(msg, attr, None)
    if vals is None or len(vals) == 0:
        return [0.0] * len(names)
    if msg.name and len(msg.name) == len(vals):
        name_to_val = {n: v for n, v in zip(msg.name, vals)}
        return [float(name_to_val.get(j, 0.0)) for j in names]
    if len(vals) < len(names):
        out = [float(v) for v in vals[: len(names)]]
        while len(out) < len(names):
            out.append(0.0)
        return out
    return [float(v) for v in vals[: len(names)]]


class FR3LinkerL6Follower(Robot):
    """FR3 arm + Linker L6 hand follower using ROS2 topics."""

    config_class = FR3LinkerL6FollowerConfig
    name = "fr3_linker_l6_follower"

    def __init__(self, config: FR3LinkerL6FollowerConfig):
        super().__init__(config)
        self.config = config
        self._node = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._connected = False
        self._owns_rclpy = False
        self._lock = threading.Lock()
        self._arm_state_msg: Optional[JointState] = None
        self._hand_state_msg: Optional[JointState] = None
        self._arm_pub = None
        self._hand_pub = None

        self._cmd_debug_counter = 0
        self._last_published_arm_pos: Optional[list[float]] = None

    @property
    def observation_features(self) -> dict[str, type]:
        features = {}
        for j in self.config.arm_joint_names:
            features[f"{j}.pos"] = float
        for j in self.config.hand_joint_names:
            features[f"{j}.pos"] = float
        return features

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"{j}.pos": float for j in self.config.arm_joint_names}
        features.update({f"{j}.pos": float for j in self.config.hand_joint_names})
        return features

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        node_name = f"fr3_linker_l6_follower_{self.id or 'default'}"
        self._node = rclpy.create_node(node_name)

        self._arm_pub = self._node.create_publisher(
            JointTrajectory, self.config.joint_command_topic, 10
        )
        self._hand_pub = self._node.create_publisher(
            JointState, self.config.hand_control_topic, 10
        )
        self._node.create_subscription(
            JointState, self.config.joint_state_topic, self._arm_state_cb, 10
        )
        self._node.create_subscription(
            JointState, self.config.hand_state_topic, self._hand_state_cb, 10
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._connected = True
        self._cmd_debug_counter = 0
        self._last_published_arm_pos = None

        hz = float(self.config.control_hz)
        traj_dt = 1.0 / hz if hz > 0 else 1.0 / 30.0
        logger.info(
            "%s connected (arm_cmd=%s, hand_cmd=%s, arm_publish=%s, control_hz=%.1f, traj_dt=%.4fs)",
            self,
            self.config.joint_command_topic,
            self.config.hand_control_topic,
            self.config.enable_arm_publish,
            hz,
            traj_dt,
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s does not require calibration", self)

    def configure(self) -> None:
        pass

    def _arm_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._arm_state_msg = msg

    def _hand_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._hand_state_msg = msg

    def _wait_for_state(self) -> None:
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                arm_ok = self._arm_state_msg is not None
                hand_ok = self._hand_state_msg is not None
            if arm_ok and hand_ok:
                return
            time.sleep(0.01)
        missing = "arm" if not arm_ok else "hand"
        raise TimeoutError(f"Timeout waiting for {missing} state on ROS2 topics")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self._wait_for_state()
        with self._lock:
            arm_msg = self._arm_state_msg
            hand_msg = self._hand_state_msg
        if arm_msg is None or hand_msg is None:
            raise RuntimeError("Arm or hand state not available")

        arm_pos = _ordered_values(arm_msg, self.config.arm_joint_names, "position")
        hand_pos = _ordered_values(hand_msg, self.config.hand_joint_names, "position")

        obs = {}
        for i, j in enumerate(self.config.arm_joint_names):
            obs[f"{j}.pos"] = arm_pos[i]
        for i, j in enumerate(self.config.hand_joint_names):
            obs[f"{j}.pos"] = hand_pos[i]
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        raw_arm_pos = []
        for j in self.config.arm_joint_names:
            key = f"{j}.pos"
            if key not in action:
                raise ValueError(f"Missing action key: {key}")
            value = float(action[key])
            if not math.isfinite(value):
                logger.warning("Received non-finite action for %s, ignoring this frame.", key)
                return action
            raw_arm_pos.append(value)

        if not self.config.enable_arm_publish:
            hand_pos = []
            for j in self.config.hand_joint_names:
                key = f"{j}.pos"
                if key not in action:
                    raise ValueError(f"Missing action key: {key}")
                hand_pos.append(float(action[key]))
            hand_msg = JointState()
            hand_msg.header.stamp = self._node.get_clock().now().to_msg()
            hand_msg.name = list(self.config.hand_joint_names)
            hand_msg.position = hand_pos
            self._hand_pub.publish(hand_msg)
            return action

        with self._lock:
            current_arm_msg = self._arm_state_msg

        if current_arm_msg is None:
            logger.warning("No joint state received yet. Skip this arm command frame.")
            return action

        current_pos = _ordered_values(current_arm_msg, self.config.arm_joint_names, "position")

        hz = float(self.config.control_hz)
        traj_dt_s = 1.0 / hz if hz > 0 else 1.0 / 30.0

        arm_msg = JointTrajectory()
        arm_msg.header.stamp = self._node.get_clock().now().to_msg()
        arm_msg.joint_names = list(self.config.arm_joint_names)
        point = JointTrajectoryPoint()
        point.positions = raw_arm_pos
        point.time_from_start = Duration(seconds=traj_dt_s).to_msg()
        arm_msg.points = [point]
        self._arm_pub.publish(arm_msg)

        self._cmd_debug_counter += 1
        if self._cmd_debug_counter % 20 == 0:
            max_raw_to_meas = max(abs(raw_arm_pos[i] - current_pos[i]) for i in range(len(raw_arm_pos)))
            max_cmd_step = 0.0
            if self._last_published_arm_pos is not None:
                max_cmd_step = max(
                    abs(raw_arm_pos[i] - self._last_published_arm_pos[i])
                    for i in range(len(raw_arm_pos))
                )
            logger.info(
                "Send FR3 cmd(rad)=%s max|raw-meas|=%.5f max|cmd_step|=%.5f traj_dt=%.4fs",
                [round(v, 4) for v in raw_arm_pos],
                max_raw_to_meas,
                max_cmd_step,
                traj_dt_s,
            )
        self._last_published_arm_pos = list(raw_arm_pos)

        hand_pos = []
        for j in self.config.hand_joint_names:
            key = f"{j}.pos"
            if key not in action:
                raise ValueError(f"Missing action key: {key}")
            hand_pos.append(float(action[key]))

        hand_msg = JointState()
        hand_msg.header.stamp = self._node.get_clock().now().to_msg()
        hand_msg.name = list(self.config.hand_joint_names)
        hand_msg.position = hand_pos
        self._hand_pub.publish(hand_msg)

        return action

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
