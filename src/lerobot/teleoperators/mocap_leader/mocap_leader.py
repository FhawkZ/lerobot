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
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_mocap_leader import MocapLeaderConfig

logger = logging.getLogger(__name__)
try:
    import pinocchio as pin
except ImportError:
    pin = None  # type: ignore[assignment]

try:
    from lerobot.third_party.mocap_ros_py.mocap_robotapi import (
        MCPApplication,
        MCPAvatar,
        MCPEventType,
        MCPSettings,
    )
except ImportError:
    try:
        from .mocap_robotapi import MCPApplication, MCPAvatar, MCPEventType, MCPSettings
    except ImportError as exc:
        MCPApplication = MCPAvatar = MCPEventType = MCPSettings = None  # type: ignore[assignment]
        logger.warning(
            "mocap_robotapi not available: MocapLeader will not work without it (%s)", exc
        )


links_parent = {
    "Hips": "world",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightTiptoe": "RightFoot",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftTiptoe": "LeftFoot",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Neck1": "Neck",
    "Head": "Neck1",
    "Head1": "Head",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand",
    "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2",
    "RightInHandIndex": "RightHand",
    "RightHandIndex1": "RightInHandIndex",
    "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2",
    "RightInHandMiddle": "RightHand",
    "RightHandMiddle1": "RightInHandMiddle",
    "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2",
    "RightInHandRing": "RightHand",
    "RightHandRing1": "RightInHandRing",
    "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2",
    "RightInHandPinky": "RightHand",
    "RightHandPinky1": "RightInHandPinky",
    "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand",
    "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2",
    "LeftInHandIndex": "LeftHand",
    "LeftHandIndex1": "LeftInHandIndex",
    "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2",
    "LeftInHandMiddle": "LeftHand",
    "LeftHandMiddle1": "LeftInHandMiddle",
    "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2",
    "LeftInHandRing": "LeftHand",
    "LeftHandRing1": "LeftInHandRing",
    "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2",
    "LeftInHandPinky": "LeftHand",
    "LeftHandPinky1": "LeftInHandPinky",
    "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2",
}


def axis_to_ros_position(position):
    return (position[2] / 100.0, position[0] / 100.0, position[1] / 100.0)


def axis_to_ros_quaternion(rotation):
    return (rotation[3], rotation[1], rotation[2], rotation[0])


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_rotate(q, v):
    vx, vy, vz = v
    qv = (vx, vy, vz, 0.0)
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[:3]


def compose_transform(parent_pos, parent_quat, local_pos, local_quat):
    rotated = quat_rotate(parent_quat, local_pos)
    return (
        (
            parent_pos[0] + rotated[0],
            parent_pos[1] + rotated[1],
            parent_pos[2] + rotated[2],
        ),
        quat_multiply(parent_quat, local_quat),
    )


# Linker L6 hand mapping from mocap:
# - finger flex: use joint angle between (parent-base) and (child-base)
# - thumb abduction/rotation: use the angle between two planes
FINGER_BASE_ANGLE_JOINTS = [
    ("RightHandIndex3", "RightHandIndex1", "RightInHandIndex", "RightHandIndex2"),
    ("RightHandMiddle3", "RightHandMiddle1", "RightInHandMiddle", "RightHandMiddle2"),
    ("RightHandRing3", "RightHandRing1", "RightInHandRing", "RightHandRing2"),
    ("RightHandPinky3", "RightHandPinky1", "RightInHandPinky", "RightHandPinky2"),
    # thumb flex @ Thumb2: angle(Thumb1-Thumb2-Thumb3)
    ("RightHandThumb3", "RightHandThumb2", "RightHandThumb1", "RightHandThumb3"),
]
ANGLE_OPEN = math.pi
ANGLE_CLOSED = 1.55


def _calculate_plane_angle(plane1_points, plane2_points) -> float:
    """
    计算两个平面的夹角，返回 [0, pi/2]（弧度）。
    平面各由 3 个点定义。
    """
    v1_1 = plane1_points[1] - plane1_points[0]
    v1_2 = plane1_points[2] - plane1_points[0]
    normal1 = np.cross(v1_1, v1_2)
    normal1 = normal1 / (np.linalg.norm(normal1) + 1e-10)

    v2_1 = plane2_points[1] - plane2_points[0]
    v2_2 = plane2_points[2] - plane2_points[0]
    normal2 = np.cross(v2_1, v2_2)
    normal2 = normal2 / (np.linalg.norm(normal2) + 1e-10)

    cos_angle = np.clip(float(np.dot(normal1, normal2)), -1.0, 1.0)
    angle = math.acos(cos_angle)
    return min(angle, math.pi - angle)


def build_local_transforms(avatar) -> dict:
    """Local transforms with axis_to_ros conversion (meters) for get_global_transform (arm pose)."""
    transforms = {}
    joints = avatar.get_joints()
    for joint in joints:
        name = joint.get_name()
        lp = joint.get_local_position()
        if lp is None:
            lp = (0.0, 0.0, 0.0)
        position = axis_to_ros_position(lp)
        rotation = axis_to_ros_quaternion(joint.get_local_rotation())
        transforms[name] = (position, rotation)
    return transforms


def _build_local_transforms(avatar: MCPAvatar) -> dict:
    local_transforms = {}
    for joint in avatar.get_joints():
        name = joint.get_name()
        lp = joint.get_local_position()
        if lp is None:
            lp = (0.0, 0.0, 0.0)
        lr = joint.get_local_rotation()  # (w,x,y,z)
        local_transforms[name] = (lp, lr)
    return local_transforms


def get_global_transform(name, local_transforms, cache):
    if name in cache:
        return cache[name]
    if name not in local_transforms:
        return None
    parent = links_parent.get(name, "world")
    if parent == "world":
        cache[name] = local_transforms[name]
        return cache[name]
    parent_tf = get_global_transform(parent, local_transforms, cache)
    if parent_tf is None:
        return None
    cache[name] = compose_transform(
        parent_tf[0], parent_tf[1], local_transforms[name][0], local_transforms[name][1]
    )
    return cache[name]


def pose_to_mat(position, quaternion_xyzw) -> np.ndarray:
    x, y, z = position
    qx, qy, qz, qw = quaternion_xyzw
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T

def _build_right_hand_relative_transforms(avatar: MCPAvatar, local_transforms: dict) -> dict:
    """
    非递归计算：从 RightHand 出发，迭代得到所有后代 joint 的相对位姿。
    返回:
      {joint_name: ((x_cm, y_cm, z_cm), (w, x, y, z))}
    """
    children: dict[str, list[str]] = {}
    parent_of: dict[str, str] = {}
    for joint in avatar.get_joints():
        child_name = joint.get_name()
        tag = joint.get_tag()
        try:
            parent_tag = joint.get_parent_joint_tag(tag)
        except RuntimeError:
            parent_tag = -1
        if parent_tag < 0 or parent_tag >= 60:
            continue
        parent_name = joint.get_name_by_tag(parent_tag)
        children.setdefault(parent_name, []).append(child_name)
        parent_of[child_name] = parent_name

    def _compose(parent_tf, local_tf):
        ppos = np.array(parent_tf[0], dtype=np.float64)
        pq = np.array([parent_tf[1][1], parent_tf[1][2], parent_tf[1][3], parent_tf[1][0]], dtype=np.float64)
        pr = Rotation.from_quat(pq)
        lpos = np.array(local_tf[0], dtype=np.float64)
        lq = np.array([local_tf[1][1], local_tf[1][2], local_tf[1][3], local_tf[1][0]], dtype=np.float64)
        lr = Rotation.from_quat(lq)
        cpos = ppos + pr.apply(lpos)
        cr = pr * lr
        cq = cr.as_quat()
        return (
            (float(cpos[0]), float(cpos[1]), float(cpos[2])),
            (float(cq[3]), float(cq[0]), float(cq[1]), float(cq[2])),
        )

    # 1) 先算 Hips->RightHand（沿父链向上累乘）
    right_hand_in_hips = None
    cur = "RightHand"
    chain = []
    while cur in parent_of and cur != "Hips":
        chain.append(cur)
        cur = parent_of[cur]
    if cur == "Hips":
        # 从 Hips 开始往下乘
        tf = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        for name in reversed(chain):
            local_tf = local_transforms.get(name)
            if local_tf is None:
                tf = None
                break
            tf = _compose(tf, local_tf)
        right_hand_in_hips = tf

    # 2) 再算 RightHand 子树上所有 joint 相对 RightHand 的位姿（保持原逻辑）
    rel: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
        # 按你的要求：RightHand 存 Hips->RightHand
        "RightHand": right_hand_in_hips or ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        # 内部计算用的 RightHand 原点（其余都相对它）
        "_RightHandOrigin": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    }
    queue = deque(["RightHand"])

    while queue:
        parent_name = queue.popleft()
        # 注意：内部 FK 仍以 RightHand 原点为根
        parent_tf = rel.get("_RightHandOrigin" if parent_name == "RightHand" else parent_name)
        if parent_tf is None:
            continue
        parent_pos_cm = np.array(parent_tf[0], dtype=np.float64)
        parent_q_xyzw = np.array(
            [parent_tf[1][1], parent_tf[1][2], parent_tf[1][3], parent_tf[1][0]],
            dtype=np.float64,
        )
        parent_rot = Rotation.from_quat(parent_q_xyzw)

        for child_name in children.get(parent_name, []):
            child_local = local_transforms.get(child_name)
            if child_local is None:
                continue
            child_pos_local = np.array(child_local[0], dtype=np.float64)
            child_q_local_xyzw = np.array(
                [child_local[1][1], child_local[1][2], child_local[1][3], child_local[1][0]],
                dtype=np.float64,
            )
            child_rot_local = Rotation.from_quat(child_q_local_xyzw)

            child_pos_rel = parent_pos_cm + parent_rot.apply(child_pos_local)
            child_rot_rel = parent_rot * child_rot_local
            child_q_rel_xyzw = child_rot_rel.as_quat()
            rel[child_name] = (
                (float(child_pos_rel[0]), float(child_pos_rel[1]), float(child_pos_rel[2])),
                (
                    float(child_q_rel_xyzw[3]),
                    float(child_q_rel_xyzw[0]),
                    float(child_q_rel_xyzw[1]),
                    float(child_q_rel_xyzw[2]),
                ),
            )
            queue.append(child_name)

    return rel

class MocapLeader(Teleoperator):
    """Mocap leader teleoperator using PNstudio skeleton + IK for FR3 + Linker L6."""

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
        self._fr3_joint_state_msg: Optional[JointState] = None
        self._prev_hand_pose: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float, float]]
        ] = None

        self._mcp_app: MCPApplication | None = None  # type: ignore[type-arg]
        self._mocap_thread: threading.Thread | None = None
        self._mocap_stop = threading.Event()

        self._latest_arm_pose: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float, float]]
        ] = None
        self._latest_finger_bends: dict[str, float] = {}

        if self.config.fr3_urdf_path:
            urdf_path = self.config.fr3_urdf_path
        else:
            urdf_path = str(Path(__file__).with_name("fr3.urdf"))
        if pin is None:
            raise ImportError(
                "pinocchio is required for MocapLeader incremental IK. "
                "Please install pinocchio in the current environment."
            )
        self._pin_model = self._build_arm_only_model(urdf_path)
        self._pin_data = self._pin_model.createData()
        self._pin_frame_id = self._choose_frame_id(self.config.fr3_ee_frame_name)
        self._last_q_deg = np.zeros(self._pin_model.nv, dtype=np.float64)
        self._ik_damp = 1e-6
        self._ik_tol = 1e-4
        # "Clearly movable" preset: increase per-cycle IK/joint delta limits.
        self._ik_max_step_norm = 0.25
        self._ik_enable_limit_avoidance = True
        self._ik_limit_threshold = 0.15
        self._ik_limit_gain = 2.0
        self._virtual_q: Optional[np.ndarray] = None
        self._max_joint_step_per_cycle = 0.04  # rad
        self._max_delta_pos_per_cycle = 0.04  # m
        self._max_delta_rot_per_cycle = 0.30  # rad
        # Motion scaling + smoothing for clearer movement and less translation jitter.
        self._delta_pos_gain = 1.8
        self._delta_rot_gain = 1.4
        self._delta_lpf_alpha = 0.7
        self._filtered_delta_x: Optional[np.ndarray] = None
        self._debug_counter = 0

        self._finger_tip_links: list[str] = [
            "RightHandThumb3",
            "RightHandIndex3",
            "RightHandMiddle3",
            "RightHandRing3",
            "RightHandPinky3",
        ]

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

        if MCPApplication is None:
            raise ImportError("mocap_robotapi is required to use MocapLeader")

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        node_name = f"mocap_leader_{self.id or 'default'}"
        self._node = rclpy.create_node(node_name)
        self._node.create_subscription(
            JointState, self.config.joint_state_topic, self._joint_state_cb, 10
        )
        self._node.create_subscription(
            JointState,
            self.config.fr3_joint_state_topic,
            self._fr3_joint_state_cb,
            10,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._mcp_app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(7012)
        settings.set_bvh_rotation(0)
        self._mcp_app.set_settings(settings)
        self._mcp_app.open()

        self._mocap_stop.clear()
        self._mocap_thread = threading.Thread(
            target=self._mocap_poll_loop, daemon=True
        )
        self._mocap_thread.start()

        self._connected = True
        logger.info("%s connected", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("%s does not require calibration", self)

    def configure(self) -> None:
        pass

    def _compute_hand_from_avatar(self, avatar: MCPAvatar) -> dict[str, float]:
        """Same hand logic as mocap_to_linkerhand MocapReader._avatar_to_action."""
        local_transforms_raw = _build_local_transforms(avatar)
        right_hand_rel = _build_right_hand_relative_transforms(
            avatar, local_transforms_raw
        )

        finger_bends: dict[str, float] = {}
        for tip_name, base_name, parent_name, child_name in FINGER_BASE_ANGLE_JOINTS:
            base_tf = right_hand_rel.get(base_name)
            parent_tf = right_hand_rel.get(parent_name)
            child_tf = right_hand_rel.get(child_name)
            if base_tf is None or parent_tf is None or child_tf is None:
                finger_bends[tip_name] = 0.0
                continue
            base_pos = np.array(base_tf[0], dtype=np.float64)
            parent_pos = np.array(parent_tf[0], dtype=np.float64)
            child_pos = np.array(child_tf[0], dtype=np.float64)
            v_parent = parent_pos - base_pos
            v_child = child_pos - base_pos
            n_p = float(np.linalg.norm(v_parent))
            n_c = float(np.linalg.norm(v_child))
            if n_p < 1e-6 or n_c < 1e-6:
                bend = 0.0
            else:
                cos_a = float(np.dot(v_parent, v_child) / (n_p * n_c))
                cos_a = max(-1.0, min(1.0, cos_a))
                angle = math.acos(cos_a)
                angle = max(ANGLE_CLOSED, min(ANGLE_OPEN, angle))
                bend = (
                    (ANGLE_OPEN - angle) / (ANGLE_OPEN - ANGLE_CLOSED)
                    if ANGLE_OPEN > ANGLE_CLOSED
                    else 0.0
                )
                bend = max(0.0, min(1.0, bend))
            finger_bends[tip_name] = bend

        thumb_tip_tf = right_hand_rel.get("RightHandThumb3")
        wrist_tf = right_hand_rel.get("_RightHandOrigin")
        mid_base_tf = right_hand_rel.get("RightInHandMiddle")
        idx_base_tf = right_hand_rel.get("RightInHandIndex")
        pky_base_tf = right_hand_rel.get("RightInHandPinky")
        thumb_rot_bend = 0.0
        if thumb_tip_tf and wrist_tf and mid_base_tf and idx_base_tf and pky_base_tf:
            thumb_fingertip = np.array(thumb_tip_tf[0], dtype=np.float64)
            wrist = np.array(wrist_tf[0], dtype=np.float64)
            joint4 = np.array(mid_base_tf[0], dtype=np.float64)
            joint5 = np.array(idx_base_tf[0], dtype=np.float64)
            joint6 = np.array(pky_base_tf[0], dtype=np.float64)
            palm_plane = [wrist, joint5, joint6]
            thumb_plane = [wrist, joint4, thumb_fingertip]
            angle = _calculate_plane_angle(palm_plane, thumb_plane)
            thumb_rot_bend = float(np.clip(angle / (math.pi / 2.0), 0.0, 1.0))
        finger_bends["_thumb_rotation"] = thumb_rot_bend
        return finger_bends

    def _mocap_poll_loop(self) -> None:
        """Background poll loop (same pattern as mocap_to_linkerhand MocapReader._poll_loop)."""
        assert self._mcp_app is not None
        period = 1.0 / self.config.mocap_poll_hz
        while not self._mocap_stop.is_set():
            t0 = time.time()

            evts = self._mcp_app.poll_next_event()
            last_avatar = None
            for evt in evts:
                if evt.event_type == MCPEventType.AvatarUpdated:
                    last_avatar = MCPAvatar(evt.event_data.avatar_handle)

            if last_avatar is not None:
                local_transforms = build_local_transforms(last_avatar)
                cache: dict[str, tuple] = {}
                right_hand_tf = get_global_transform(
                    "RightHand", local_transforms, cache
                )
                if right_hand_tf is not None:
                    hand_pos, hand_quat = right_hand_tf
                    finger_bends = self._compute_hand_from_avatar(last_avatar)
                    with self._lock:
                        self._latest_arm_pose = (hand_pos, hand_quat)
                        self._latest_finger_bends = finger_bends

            elapsed = time.time() - t0
            remain = period - elapsed
            if remain > 0:
                time.sleep(remain)

    def _joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._joint_state_msg = msg

    def _fr3_joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._fr3_joint_state_msg = msg

    def _ordered_arm_positions_rad(self) -> list[float]:
        """Extract FR3 joint positions (radians) from measured_joint_states."""
        with self._lock:
            msg = self._fr3_joint_state_msg
        if msg is None:
            raise RuntimeError("FR3 joint state not available")
        if msg.name and len(msg.name) == len(msg.position):
            name_to_pos = {n: float(p) for n, p in zip(msg.name, msg.position)}
            return [name_to_pos.get(j, 0.0) for j in self.config.arm_joint_names]
        if len(msg.position) < len(self.config.arm_joint_names):
            raise ValueError("FR3 JointState has fewer positions than arm joints")
        return [float(msg.position[i]) for i in range(len(self.config.arm_joint_names))]

    def _build_arm_only_model(self, urdf_path: str):
        assert pin is not None
        full_model = pin.buildModelFromUrdf(urdf_path)
        joint_names_to_lock = {"fr3_finger_joint1", "fr3_finger_joint2"}
        joints_to_lock = []
        for joint_name in joint_names_to_lock:
            if full_model.existJointName(joint_name):
                joints_to_lock.append(full_model.getJointId(joint_name))
        if not joints_to_lock:
            return full_model
        q_ref = pin.neutral(full_model)
        return pin.buildReducedModel(full_model, joints_to_lock, q_ref)

    def _choose_frame_id(self, preferred_frame: str) -> int:
        assert pin is not None
        if self._pin_model.existFrame(preferred_frame):
            return self._pin_model.getFrameId(preferred_frame)
        for frame_name in ["fr3_link8", "fr3_hand", "fr3_hand_tcp"]:
            if self._pin_model.existFrame(frame_name):
                return self._pin_model.getFrameId(frame_name)
        return self._pin_model.nframes - 1

    def _compute_incremental_ik(self, q_curr: np.ndarray, delta_x: np.ndarray) -> np.ndarray:
        assert pin is not None
        model = self._pin_model
        data = self._pin_data
        frame_id = self._pin_frame_id
        t0 = time.perf_counter()
        q = q_curr.copy()
        q_min = model.lowerPositionLimit
        q_max = model.upperPositionLimit

        # 目标定义：在当前位姿上叠加 delta_x
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        o_m_f0 = data.oMf[frame_id]
        target_t = o_m_f0.translation + delta_x[:3]
        target_r = pin.exp3(delta_x[3:]) @ o_m_f0.rotation

        jacobian = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        jj_t = jacobian @ jacobian.T
        h_damped = jj_t + self._ik_damp * np.eye(6)
        h_inv = np.linalg.solve(h_damped, np.eye(6))
        j_pinv = jacobian.T @ h_inv
        delta_q_task = j_pinv @ delta_x

        delta_q = delta_q_task

        if self._ik_enable_limit_avoidance:
            v_redundant = np.zeros(model.nv)
            for j in range(model.nv):
                dist_to_lower = q[j] - q_min[j]
                dist_to_upper = q_max[j] - q[j]
                if dist_to_lower < self._ik_limit_threshold:
                    v_redundant[j] = self._ik_limit_gain * (
                        self._ik_limit_threshold - dist_to_lower
                    ) ** 2
                elif dist_to_upper < self._ik_limit_threshold:
                    v_redundant[j] = -self._ik_limit_gain * (
                        self._ik_limit_threshold - dist_to_upper
                    ) ** 2
            p_null = np.eye(model.nv) - j_pinv @ jacobian
            delta_q += p_null @ v_redundant

        step_norm = float(np.linalg.norm(delta_q))
        if step_norm > self._ik_max_step_norm and step_norm > 1e-12:
            delta_q *= self._ik_max_step_norm / step_norm

        q_next = np.clip(q + delta_q, q_min, q_max)

        # 后验误差统计（对齐 fr3_incremental_ik.py）
        pos_err0 = target_t - o_m_f0.translation
        rot_err0 = pin.log3(target_r @ o_m_f0.rotation.T)
        initial_err_norm = float(np.linalg.norm(np.hstack([pos_err0, rot_err0])))

        pin.forwardKinematics(model, data, q_next)
        pin.updateFramePlacements(model, data)
        o_m_f1 = data.oMf[frame_id]
        pos_err1 = target_t - o_m_f1.translation
        rot_err1 = pin.log3(target_r @ o_m_f1.rotation.T)
        final_err_norm = float(np.linalg.norm(np.hstack([pos_err1, rot_err1])))
        if final_err_norm > self._ik_tol:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "Incremental IK residual final=%.6f initial=%.6f elapsed_ms=%.3f",
                final_err_norm,
                initial_err_norm,
                elapsed_ms,
            )

        return q_next - q_curr

    def reset_incremental_pose(self) -> None:
        """Reset incremental state to avoid first-frame jumps across episode boundaries."""
        with self._lock:
            self._virtual_q = None
            self._filtered_delta_x = None
            if self._latest_arm_pose is not None:
                hand_pos, hand_quat = self._latest_arm_pose
                self._prev_hand_pose = (
                    (hand_pos[0], hand_pos[1], hand_pos[2]),
                    (hand_quat[0], hand_quat[1], hand_quat[2], hand_quat[3]),
                )
            else:
                self._prev_hand_pose = None

    def _wait_for_mocap(self) -> None:
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_arm_pose is not None:
                    return
            time.sleep(0.01)
        raise TimeoutError("Timeout waiting for mocap data from PNstudio")

    def _wait_for_fr3_joint_state(self) -> None:
        deadline = time.monotonic() + self.config.timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._fr3_joint_state_msg is not None:
                    return
            time.sleep(0.01)
        raise TimeoutError(
            f"Timeout waiting for FR3 joint state on {self.config.fr3_joint_state_topic}"
        )

    def _compute_fr3_joints(self) -> list[float]:
        self._wait_for_mocap()
        self._wait_for_fr3_joint_state()

        with self._lock:
            assert self._latest_arm_pose is not None
            hand_pos, hand_quat = self._latest_arm_pose

        arm_pos_rad = self._ordered_arm_positions_rad()
        q_curr = np.array(arm_pos_rad, dtype=np.float64)

        if self._prev_hand_pose is None:
            delta_pos = (0.0, 0.0, 0.0)
            delta_quat = (0.0, 0.0, 0.0, 1.0)
        else:
            prev_pos, prev_quat = self._prev_hand_pose
            delta_pos = (
                hand_pos[0] - prev_pos[0],
                hand_pos[1] - prev_pos[1],
                hand_pos[2] - prev_pos[2],
            )
            delta_quat = quat_multiply(hand_quat, quat_conjugate(prev_quat))

        with self._lock:
            self._prev_hand_pose = (hand_pos, hand_quat)

        delta_rotvec = Rotation.from_quat(np.array(delta_quat, dtype=np.float64)).as_rotvec()
        delta_pos_arr = np.array(delta_pos, dtype=np.float64)

        if self.config.enable_mocap_to_fr3_axis_mapping:
            # Mapping:
            #   X_robot = Y_mocap
            #   Y_robot = -X_mocap
            #   Z_robot = Z_mocap
            aligned_delta_pos = np.array(
                [delta_pos_arr[1], -delta_pos_arr[0], delta_pos_arr[2]],
                dtype=np.float64,
            )
            aligned_delta_rotvec = np.array(
                [delta_rotvec[1], -delta_rotvec[0], delta_rotvec[2]],
                dtype=np.float64,
            )
            delta_x = np.hstack([aligned_delta_pos, aligned_delta_rotvec])
        else:
            delta_x = np.hstack([delta_pos_arr, delta_rotvec])

        # Scale mocap deltas to better match human hand motion magnitude.
        delta_x[:3] *= self._delta_pos_gain
        delta_x[3:] *= self._delta_rot_gain

        # Low-pass delta stream to reduce jitter/stutter in translation.
        if self._filtered_delta_x is None:
            self._filtered_delta_x = delta_x.copy()
        else:
            self._filtered_delta_x = (
                self._delta_lpf_alpha * delta_x
                + (1.0 - self._delta_lpf_alpha) * self._filtered_delta_x
            )
        delta_x = self._filtered_delta_x.copy()

        # Clamp mocap glitches to keep the robot command stream continuous.
        delta_pos_norm = float(np.linalg.norm(delta_x[:3]))
        if delta_pos_norm > self._max_delta_pos_per_cycle and delta_pos_norm > 1e-12:
            delta_x[:3] *= self._max_delta_pos_per_cycle / delta_pos_norm
        delta_rot_norm = float(np.linalg.norm(delta_x[3:]))
        if delta_rot_norm > self._max_delta_rot_per_cycle and delta_rot_norm > 1e-12:
            delta_x[3:] *= self._max_delta_rot_per_cycle / delta_rot_norm

        delta_q = self._compute_incremental_ik(q_curr=q_curr, delta_x=delta_x)
        delta_q = np.clip(
            delta_q,
            -self._max_joint_step_per_cycle,
            self._max_joint_step_per_cycle,
        )
        q_next = q_curr + delta_q
        self._last_q_deg = np.rad2deg(q_next)
        self._debug_counter += 1
        if self._debug_counter % 30 == 0:
            logger.info(
                "Mocap->FR3 delta pos=%.4f m rot=%.4f rad q(rad)=%s",
                float(np.linalg.norm(delta_x[:3])),
                float(np.linalg.norm(delta_x[3:])),
                [round(float(v), 4) for v in q_next[: len(self.config.arm_joint_names)]],
            )
        return [float(v) for v in q_next[: len(self.config.arm_joint_names)]]

    def _compute_hand_joints(self) -> list[float]:
        with self._lock:
            finger_bends = dict(self._latest_finger_bends)

        # mocap_to_linkerhand convention:
        # - bend v01: 0=open, 1=closed
        # - LinkerHand L6 position: 0=closed, 255=open
        def _to_u8_open(v01: float) -> float:
            v01 = float(np.clip(v01, 0.0, 1.0))
            return float(255.0 * (1.0 - v01))

        thumb_flex = float(finger_bends.get("RightHandThumb3", 0.0))
        thumb_abd = float(finger_bends.get("_thumb_rotation", 0.0))
        index_flex = float(finger_bends.get("RightHandIndex3", 0.0))
        middle_flex = float(finger_bends.get("RightHandMiddle3", 0.0))
        ring_flex = float(finger_bends.get("RightHandRing3", 0.0))
        little_flex = float(finger_bends.get("RightHandPinky3", 0.0))

        joints = [
            _to_u8_open(thumb_flex),
            _to_u8_open(thumb_abd),
            _to_u8_open(index_flex),
            _to_u8_open(middle_flex),
            _to_u8_open(ring_flex),
            _to_u8_open(little_flex),
        ]
        return joints[: len(self.config.hand_joint_names)]

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        arm_pos = self._compute_fr3_joints()
        hand_pos = self._compute_hand_joints()

        action: dict[str, float] = {}
        for idx, joint in enumerate(self.config.arm_joint_names):
            action[f"{joint}.pos"] = arm_pos[idx]
        for idx, joint in enumerate(self.config.hand_joint_names):
            if idx < len(hand_pos):
                action[f"{joint}.pos"] = hand_pos[idx]
            else:
                action[f"{joint}.pos"] = 0.0
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self._mocap_stop.set()
        if self._mocap_thread is not None:
            self._mocap_thread.join(timeout=1.0)
            self._mocap_thread = None
        if self._mcp_app is not None:
            try:
                self._mcp_app.close()
            except Exception:
                pass
            self._mcp_app = None

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
