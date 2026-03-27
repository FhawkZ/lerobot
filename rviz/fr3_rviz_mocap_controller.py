#!/usr/bin/env python3

import argparse
import math
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from lerobot.model.kinematics import RobotKinematics

try:
    from lerobot.third_party.mocap_ros_py.mocap_robotapi import (
        MCPApplication,
        MCPAvatar,
        MCPEventType,
        MCPSettings,
    )
except ImportError:
    from lerobot.teleoperators.mocap_leader.mocap_robotapi import (
        MCPApplication,
        MCPAvatar,
        MCPEventType,
        MCPSettings,
    )


LINKS_PARENT = {
    "Hips": "world",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
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


def build_local_transforms(avatar) -> dict:
    transforms = {}
    for joint in avatar.get_joints():
        name = joint.get_name()
        lp = joint.get_local_position() or (0.0, 0.0, 0.0)
        transforms[name] = (axis_to_ros_position(lp), axis_to_ros_quaternion(joint.get_local_rotation()))
    return transforms


def get_global_transform(name, local_transforms, cache):
    if name in cache:
        return cache[name]
    if name not in local_transforms:
        return None
    parent = LINKS_PARENT.get(name, "world")
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

    r_mat = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r_mat
    out[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return out


class Fr3RvizMocapController:
    def __init__(
        self,
        urdf_path: str,
        ee_frame_name: str,
        arm_joint_names: list[str],
        publish_topic: str,
        initial_joint_state_topic: str,
        mocap_udp_port: int,
        mocap_poll_hz: float,
        control_hz: float,
    ):
        self._arm_joint_names = arm_joint_names
        self._publish_topic = publish_topic
        self._initial_joint_state_topic = initial_joint_state_topic
        self._mocap_poll_hz = mocap_poll_hz
        self._control_hz = control_hz

        self._node = rclpy.create_node("fr3_rviz_mocap_controller")
        self._publisher = self._node.create_publisher(JointState, publish_topic, 10)
        self._joint_state_msg: Optional[JointState] = None
        self._lock = threading.Lock()

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        if initial_joint_state_topic:
            self._node.create_subscription(
                JointState, initial_joint_state_topic, self._joint_state_cb, 10
            )

        self._ik = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=ee_frame_name,
            joint_names=list(arm_joint_names),
        )
        self._q_deg = np.zeros(len(arm_joint_names), dtype=np.float64)
        self._latest_hand_pose: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float, float]]
        ] = None
        self._prev_hand_pose: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float, float]]
        ] = None

        self._mcp_app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(mocap_udp_port)
        settings.set_bvh_rotation(0)
        self._mcp_app.set_settings(settings)
        self._mcp_app.open()

        self._mocap_stop = threading.Event()
        self._mocap_thread = threading.Thread(target=self._mocap_poll_loop, daemon=True)
        self._mocap_thread.start()

    def _joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._joint_state_msg = msg

    def _ordered_positions_rad(self) -> Optional[list[float]]:
        with self._lock:
            msg = self._joint_state_msg
        if msg is None:
            return None
        if msg.name and len(msg.name) == len(msg.position):
            name_to_pos = {n: float(p) for n, p in zip(msg.name, msg.position)}
            return [name_to_pos.get(j, 0.0) for j in self._arm_joint_names]
        if len(msg.position) >= len(self._arm_joint_names):
            return [float(msg.position[i]) for i in range(len(self._arm_joint_names))]
        return None

    def _mocap_poll_loop(self) -> None:
        period = 1.0 / self._mocap_poll_hz
        while not self._mocap_stop.is_set():
            t0 = time.time()
            evts = self._mcp_app.poll_next_event()
            last_avatar = None
            for evt in evts:
                if evt.event_type == MCPEventType.AvatarUpdated:
                    last_avatar = MCPAvatar(evt.event_data.avatar_handle)

            if last_avatar is not None:
                local_transforms = build_local_transforms(last_avatar)
                cache = {}
                right_hand_tf = get_global_transform("RightHand", local_transforms, cache)
                if right_hand_tf is not None:
                    with self._lock:
                        self._latest_hand_pose = right_hand_tf

            remain = period - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)

    def _compute_q_rad(self) -> list[float]:
        with self._lock:
            hand_pose = self._latest_hand_pose
        if hand_pose is None:
            return [float(v) for v in np.deg2rad(self._q_deg)]

        current_measured = self._ordered_positions_rad()
        if current_measured is not None:
            self._q_deg = np.rad2deg(np.asarray(current_measured, dtype=np.float64))

        hand_pos, hand_quat = hand_pose
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
        self._prev_hand_pose = (hand_pos, hand_quat)

        t_ee_current = self._ik.forward_kinematics(self._q_deg.copy())
        current_pos = t_ee_current[:3, 3].copy()
        current_r = t_ee_current[:3, :3].copy()
        delta_r = pose_to_mat((0.0, 0.0, 0.0), delta_quat)[:3, :3]
        target_pos = current_pos + np.asarray(delta_pos, dtype=np.float64)
        target_r = delta_r @ current_r

        t_target = np.eye(4, dtype=np.float64)
        t_target[:3, :3] = target_r
        t_target[:3, 3] = target_pos

        self._q_deg = self._ik.inverse_kinematics(
            current_joint_pos=self._q_deg.copy(),
            desired_ee_pose=t_target,
            position_weight=1.0,
            orientation_weight=0.1,
        )
        q_rad = np.deg2rad(self._q_deg[: len(self._arm_joint_names)])
        return [float(v) for v in q_rad]

    def run(self) -> None:
        period = 1.0 / self._control_hz
        self._node.get_logger().info(
            f"Running mocap->FR3 RViz controller, publish_topic={self._publish_topic}"
        )
        try:
            while rclpy.ok():
                t0 = time.time()
                q_rad = self._compute_q_rad()
                msg = JointState()
                msg.header.stamp = self._node.get_clock().now().to_msg()
                msg.name = list(self._arm_joint_names)
                msg.position = q_rad
                self._publisher.publish(msg)

                remain = period - (time.time() - t0)
                if remain > 0:
                    time.sleep(remain)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._mocap_stop.set()
        if self._mocap_thread is not None:
            self._mocap_thread.join(timeout=1.0)
        if self._mcp_app is not None:
            try:
                self._mcp_app.close()
            except Exception:
                pass
        if self._executor and self._node:
            self._executor.remove_node(self._node)
            self._executor.shutdown()
        if self._node:
            self._node.destroy_node()


def _default_urdf_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "lerobot"
        / "teleoperators"
        / "mocap_leader"
        / "fr3.urdf"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Mocap controls FR3 in RViz via /joint_states")
    parser.add_argument("--urdf-path", default=_default_urdf_path(), help="FR3 URDF file path")
    parser.add_argument("--ee-frame-name", default="fr3_link8")
    parser.add_argument("--publish-topic", default="/joint_states")
    parser.add_argument(
        "--initial-joint-state-topic",
        default="",
        help="Optional external joint state topic for IK warm start",
    )
    parser.add_argument("--mocap-udp-port", type=int, default=7012)
    parser.add_argument("--mocap-poll-hz", type=float, default=120.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    arm_joint_names = [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]
    controller = Fr3RvizMocapController(
        urdf_path=args.urdf_path,
        ee_frame_name=args.ee_frame_name,
        arm_joint_names=arm_joint_names,
        publish_topic=args.publish_topic,
        initial_joint_state_topic=args.initial_joint_state_topic,
        mocap_udp_port=args.mocap_udp_port,
        mocap_poll_hz=args.mocap_poll_hz,
        control_hz=args.control_hz,
    )
    controller.run()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
