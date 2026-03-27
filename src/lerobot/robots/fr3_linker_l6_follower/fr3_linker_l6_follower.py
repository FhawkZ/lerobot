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

        # ================= 🟢 核心保险层：变量初始化 =================
        # 用于 EMA 低通滤波的状态记忆
        self._ema_arm_pos: Optional[list[float]] = None
        # 平滑系数 (0~1)。适当减小可显著降低关节速度/加速度不连续风险。
        self._alpha = 0.2
        # 安全截断：限制单步最大变化量（弧度），抑制突变触发 Franka reflex。
        # 在 60Hz 下，0.01 rad/step ≈ 0.6 rad/s。
        self._max_dq_per_step = 0.02
        # 以“上一次已发送命令”为基准再做一次硬限幅，避免状态抖动导致突变。
        self._last_cmd_arm_pos: Optional[list[float]] = None
        self._last_cmd_step: Optional[list[float]] = None
        # 限制每一帧“步长变化”，等效限制加速度突变。
        self._max_ddq_per_step = 0.002
        # 连续多帧观测到大跳变再重置 EMA，避免单帧噪声触发反复重置。
        self._jump_count = 0
        self._jump_reset_threshold = 10
        self._cmd_debug_counter = 0
        self._startup_hold_s = 0.5
        self._publish_min_interval_s = 1.0 / 30.0  # 30 Hz
        self._traj_time_from_start_s = 0.08
        self._connect_monotonic_s: Optional[float] = None
        self._last_publish_monotonic_s: Optional[float] = None
        self._logged_startup_hold = False
        self._last_measured_arm_pos: Optional[list[float]] = None
        self._motion_gate_threshold_rad = 0.0
        self._max_cmd_to_meas_rad = 0.02
        self._arm_publish_locked = True
        self._unlock_motion_threshold_rad = 0.004
        self._unlock_consecutive_frames = 2
        self._unlock_counter = 0
        self._lock_debug_counter = 0
        self._unlock_blend = 0.0
        self._unlock_blend_step = 0.1  # ~1s to fully open at 10Hz
        # ==========================================================

    @property
    def observation_features(self) -> dict[str, type]:
        features = {}
        for j in self.config.arm_joint_names:
            features[f"{j}.pos"] = float
            # features[f"{j}.vel"] = float
            # features[f"{j}.torque"] = float
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
        self._connect_monotonic_s = time.monotonic()
        self._last_publish_monotonic_s = None
        self._logged_startup_hold = False
        self._arm_publish_locked = True
        self._unlock_counter = 0
        self._lock_debug_counter = 0
        self._unlock_blend = 0.0
        logger.info(
            "%s connected (arm_cmd=%s, hand_cmd=%s, arm_publish=%s, hold_s=%.2f, publish_hz=%.1f, traj_t=%.2f, arm_locked=%s)",
            self,
            self.config.joint_command_topic,
            self.config.hand_control_topic,
            self.config.enable_arm_publish,
            self._startup_hold_s,
            1.0 / self._publish_min_interval_s,
            self._traj_time_from_start_s,
            self._arm_publish_locked,
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
        # arm_vel = _ordered_values(arm_msg, self.config.arm_joint_names, "velocity")
        # arm_eff = _ordered_values(arm_msg, self.config.arm_joint_names, "effort")
        hand_pos = _ordered_values(hand_msg, self.config.hand_joint_names, "position")
        # Keep observation fields aligned with `observation_features`:
        # only hand position is exposed for now.
        # hand_vel = _ordered_values(hand_msg, self.config.hand_joint_names, "velocity")
        # hand_eff = _ordered_values(hand_msg, self.config.hand_joint_names, "effort")

        obs = {}
        for i, j in enumerate(self.config.arm_joint_names):
            obs[f"{j}.pos"] = arm_pos[i]
            # obs[f"{j}.vel"] = arm_vel[i]
            # obs[f"{j}.torque"] = arm_eff[i]
        for i, j in enumerate(self.config.hand_joint_names):
            obs[f"{j}.pos"] = hand_pos[i]
            # obs[f"{j}.vel"] = hand_vel[i]
            # obs[f"{j}.torque"] = hand_eff[i]
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
        else:
            current_pos = _ordered_values(current_arm_msg, self.config.arm_joint_names, "position")
            self._last_measured_arm_pos = current_pos.copy()
            # Remove follower-side suppression: pass leader target through directly.
            safe_arm_pos = raw_arm_pos.copy()

        now_s = time.monotonic()
        dt_since_connect = (
            now_s - self._connect_monotonic_s if self._connect_monotonic_s is not None else None
        )
        if dt_since_connect is not None and dt_since_connect < self._startup_hold_s:
            # Startup protection: keep command pinned to measured pose for a short period.
            safe_arm_pos = current_pos.copy()
            self._ema_arm_pos = current_pos.copy()
            self._last_cmd_step = [0.0] * len(safe_arm_pos)
            self._last_cmd_arm_pos = safe_arm_pos.copy()
            if not self._logged_startup_hold:
                logger.warning(
                    "FR3 startup hold enabled for %.2fs: publishing measured joints only.",
                    self._startup_hold_s,
                )
                self._logged_startup_hold = True
        else:
            self._arm_publish_locked = False

        if self._last_publish_monotonic_s is not None:
            dt_pub = now_s - self._last_publish_monotonic_s
            if dt_pub < self._publish_min_interval_s:
                return action

        arm_msg = JointTrajectory()
        arm_msg.header.stamp = self._node.get_clock().now().to_msg()
        arm_msg.joint_names = list(self.config.arm_joint_names)
        point = JointTrajectoryPoint()
        point.positions = safe_arm_pos
        point.time_from_start = Duration(seconds=self._traj_time_from_start_s).to_msg()
        arm_msg.points = [point]
        self._arm_pub.publish(arm_msg)
        self._last_publish_monotonic_s = now_s
        if self._last_cmd_arm_pos is None:
            self._last_cmd_step = [0.0] * len(safe_arm_pos)
        else:
            self._last_cmd_step = [
                safe_arm_pos[i] - self._last_cmd_arm_pos[i] for i in range(len(safe_arm_pos))
            ]
        self._last_cmd_arm_pos = safe_arm_pos.copy()
        self._cmd_debug_counter += 1
        if self._cmd_debug_counter % 20 == 0:
            max_raw_to_meas = max(abs(raw_arm_pos[i] - current_pos[i]) for i in range(len(raw_arm_pos)))
            max_cmd_to_meas = max(abs(safe_arm_pos[i] - current_pos[i]) for i in range(len(raw_arm_pos)))
            max_cmd_step = (
                max(abs(v) for v in self._last_cmd_step) if self._last_cmd_step is not None else 0.0
            )
            logger.info(
                "Send FR3 cmd(rad)=%s (raw=%s) max|raw-meas|=%.5f max|cmd-meas|=%.5f max|cmd_step|=%.5f dt_connect=%.3f",
                [round(v, 4) for v in safe_arm_pos],
                [round(v, 4) for v in raw_arm_pos],
                max_raw_to_meas,
                max_cmd_to_meas,
                max_cmd_step,
                -1.0 if dt_since_connect is None else dt_since_connect,
            )

        # 4. 手爪控制逻辑：直接透传 action 值
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
