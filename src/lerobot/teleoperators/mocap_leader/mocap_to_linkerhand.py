"""
主循环版 mocap -> LinkerHandL6 控制桥接脚本（无 ROS 定时器）。

符合以下流程：
1) connect 两个类：
   - MocapReader: 高频读取并处理 mocap 数据
   - LinkerHandL6Robot: 通过 ROS topic 发送 LinkerHandL6 控制指令
2) 主循环：
   - 从 MocapReader 读取最新 action
   - 写入 LinkerHandL6Robot
3) 主循环以固定频率运行（默认 60Hz），MocapReader 内部采样频率可单独调高（默认 120Hz）
"""

import math
import threading
import time
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from .mocap_robotapi import MCPApplication, MCPAvatar, MCPEventType, MCPSettings
from collections import deque

FINGER_BASE_ANGLE_JOINTS = [
    ("RightHandIndex3", "RightHandIndex1", "RightInHandIndex", "RightHandIndex2"),
    ("RightHandMiddle3", "RightHandMiddle1", "RightInHandMiddle", "RightHandMiddle2"),
    ("RightHandRing3", "RightHandRing1", "RightInHandRing", "RightHandRing2"),
    ("RightHandPinky3", "RightHandPinky1", "RightInHandPinky", "RightHandPinky2"),
    # thumb bend @Thumb2: angle(Thumb1-Thumb2-Thumb3)
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


class MocapReader:
    """高频读取并处理 mocap 数据，产出 LinkerHandL6 action（6 维关节角）。"""

    def __init__(self, udp_port: int = 7012, poll_hz: float = 60.0):
        self.udp_port = udp_port
        self.poll_hz = poll_hz
        self._app: Optional[MCPApplication] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_action: Optional[Dict[str, float]] = None
        self._last_log_t = time.time()
        self._frame_cnt = 0
        self._perf_rel_ms_sum = 0.0
        self._perf_rel_ms_cnt = 0

    def connect(self) -> None:
        self._app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(self.udp_port)
        settings.set_bvh_rotation(0)
        self._app.set_settings(settings)
        ok, msg = self._app.open()
        if not ok:
            raise RuntimeError(f"打开 MCPApplication 失败: {msg}")

        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[MocapReader] connected, udp={self.udp_port}, poll_hz={self.poll_hz}")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._app is not None:
            self._app.close()
            self._app = None

    def get_action(self) -> Optional[Dict[str, float]]:
        with self._lock:
            if self._latest_action is None:
                return None
            return dict(self._latest_action)

    def _poll_loop(self) -> None:
        assert self._app is not None
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            t0 = time.time()

            evts = self._app.poll_next_event()
            last_avatar = None
            for evt in evts:
                if evt.event_type == MCPEventType.AvatarUpdated:
                    last_avatar = MCPAvatar(evt.event_data.avatar_handle)

            if last_avatar is not None:
                action = self._avatar_to_action(last_avatar)
                if action is not None:
                    with self._lock:
                        self._latest_action = action
                    self._frame_cnt += 1

            now = time.time()
            if now - self._last_log_t >= 1.0:
                dt = now - self._last_log_t
                fps = self._frame_cnt / dt if dt > 0 else 0.0
                avg_rel = (self._perf_rel_ms_sum / self._perf_rel_ms_cnt) if self._perf_rel_ms_cnt else 0.0
                print(f"[MocapReader] action update ~ {fps:.1f} Hz | rel_tf avg {avg_rel:.3f} ms")
                self._frame_cnt = 0
                self._last_log_t = now
                self._perf_rel_ms_sum = 0.0
                self._perf_rel_ms_cnt = 0

            elapsed = time.time() - t0
            remain = period - elapsed
            if remain > 0:
                time.sleep(remain)

    def _avatar_to_action(self, avatar: MCPAvatar) -> Optional[Dict[str, float]]:
        local_transforms = _build_local_transforms(avatar)
        t_rel_start = time.perf_counter()
        right_hand_rel = _build_right_hand_relative_transforms(avatar, local_transforms)
        t_rel_ms = (time.perf_counter() - t_rel_start) * 1000.0
        self._perf_rel_ms_sum += t_rel_ms
        self._perf_rel_ms_cnt += 1

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
            n_p = np.linalg.norm(v_parent)
            n_c = np.linalg.norm(v_child)
            if n_p < 1e-6 or n_c < 1e-6:
                bend = 0.0
            else:
                cos_a = float(np.dot(v_parent, v_child) / (n_p * n_c))
                cos_a = max(-1.0, min(1.0, cos_a))
                angle = math.acos(cos_a)
                angle = max(ANGLE_CLOSED, min(ANGLE_OPEN, angle))
                bend = (ANGLE_OPEN - angle) / (ANGLE_OPEN - ANGLE_CLOSED) if ANGLE_OPEN > ANGLE_CLOSED else 0.0
                bend = max(0.0, min(1.0, bend))
            finger_bends[tip_name] = bend

        # 拇指旋转分量
        thumb_tip_tf = right_hand_rel.get("RightHandThumb3")
        # 其余关节仍是 RightHand 坐标系下；wrist 取 RightHand 原点
        wrist_tf = right_hand_rel.get("_RightHandOrigin")
        # 选用掌心根部点（更稳定）：RightInHandMiddle / RightInHandIndex / RightInHandPinky
        mid_base_tf = right_hand_rel.get("RightInHandMiddle")
        idx_base_tf = right_hand_rel.get("RightInHandIndex")
        pky_base_tf = right_hand_rel.get("RightInHandPinky")
        thumb_rot_bend = 0.0
        if thumb_tip_tf and wrist_tf and mid_base_tf and idx_base_tf and pky_base_tf:
            thumb_fingertip = np.array(thumb_tip_tf[0], dtype=np.float64)
            wrist = np.array(wrist_tf[0], dtype=np.float64)
            joint4 = np.array(mid_base_tf[0], dtype=np.float64)  # RightInHandMiddle
            joint5 = np.array(idx_base_tf[0], dtype=np.float64)  # RightInHandIndex
            joint6 = np.array(pky_base_tf[0], dtype=np.float64)  # RightInHandPinky

            # Plane 1 (palm plane): wrist, joint5, joint6
            palm_plane = [wrist, joint5, joint6]
            # Plane 2 (thumb rotation plane): wrist, joint4, thumb_fingertip
            thumb_plane = [wrist, joint4, thumb_fingertip]

            angle = _calculate_plane_angle(palm_plane, thumb_plane)
            thumb_rot_bend = float(np.clip(angle / (math.pi / 2.0), 0.0, 1.0))
        finger_bends["_thumb_rotation"] = thumb_rot_bend
        
        # 输出为可直接下发到 LinkerHand ROS2 SDK 的 position: [0,255]
        # 设备约定: 0=弯曲，255=张开
        def _to_u8_open(v01: float) -> float:
            v01 = float(np.clip(v01, 0.0, 1.0))
            return float(255.0 * (1.0 - v01))

        def _to_u8(v01: float) -> float:
            v01 = float(np.clip(v01, 0.0, 1.0))
            return float(255.0 * v01)

        thumb_flex = float(finger_bends.get("RightHandThumb3", 0.0))
        index_flex = float(finger_bends.get("RightHandIndex3", 0.0))
        middle_flex = float(finger_bends.get("RightHandMiddle3", 0.0))
        ring_flex = float(finger_bends.get("RightHandRing3", 0.0))
        little_flex = float(finger_bends.get("RightHandPinky3", 0.0))
        thumb_abd = float(finger_bends.get("_thumb_rotation", 0.0))

        return {
            # L6 顺序: ["Thumb Flex", "Thumb Abduction", "Index Flex", "Middle Flex", "Ring Flex", "Little Flex"]
            "hand_0": _to_u8_open(thumb_flex),
            # 拇指外展/旋转方向反向：用 1-x 翻转
            "hand_1": _to_u8_open(thumb_abd),
            "hand_2": _to_u8_open(index_flex),
            "hand_3": _to_u8_open(middle_flex),
            "hand_4": _to_u8_open(ring_flex),
            "hand_5": _to_u8_open(little_flex),
        }


class LinkerHandL6Robot:
    """通过 ROS2 topic 发布 LinkerHand L6 指令（0~255）。"""

    def __init__(
        self,
        hand_control_topic: str = "/cb_right_hand_control_cmd",
        hand_joint_names: Optional[list[str]] = None,
    ):
        self.hand_control_topic = hand_control_topic
        # L6 position 顺序（来自 linkerhand-ros2-sdk README）:
        # ["Thumb Flex", "Thumb Abduction", "Index Flex", "Middle Flex", "Ring Flex", "Little Flex"]
        self.hand_joint_names = hand_joint_names or ["hand_0", "hand_1", "hand_2", "hand_3", "hand_4", "hand_5"]
        self.node: Optional[Node] = None
        self.pub = None

    def connect(self) -> None:
        self.node = rclpy.create_node("linkerhand_l6_bridge")
        self.pub = self.node.create_publisher(JointState, self.hand_control_topic, 10)
        self.node.get_logger().info(f"LinkerHandL6Robot connected, topic={self.hand_control_topic}")

    def send_action(self, action: Dict[str, float]) -> None:
        if self.node is None or self.pub is None:
            return
        # action 已经是 [0,255] 的 LinkerHand L6 position 值，按 hand_joint_names 顺序直接发
        cmd = [float(action.get(name, 0.0)) for name in self.hand_joint_names]

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        # linkerhand ROS2 SDK 示例里 name 通常为空，位置序列按型号固定映射
        msg.name = []
        msg.position = cmd
        self.pub.publish(msg)

    def disconnect(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        self.pub = None


def main(
    mocap_poll_hz: float = 120.0,
    control_hz: float = 30.0,
    udp_port: int = 7012,
    hand_control_topic: str = "/cb_right_hand_control_cmd",
) -> None:
    """
    主循环：
      connect(mocap, robot) -> while循环读action写action -> 按 control_hz 控频。
    """
    rclpy.init()
    mocap = MocapReader(udp_port=udp_port, poll_hz=mocap_poll_hz)
    robot = LinkerHandL6Robot(hand_control_topic=hand_control_topic)
    mocap.connect()
    robot.connect()

    print(f"[MainLoop] start, control_hz={control_hz}, mocap_poll_hz={mocap_poll_hz}")
    period = 1.0 / control_hz
    frame_cnt = 0
    last_t = time.time()

    try:
        while rclpy.ok():
            t0 = time.time()
            action = mocap.get_action()
            if action is not None:
                robot.send_action(action)
                frame_cnt += 1

            # 处理 ROS 事件（非阻塞）
            if robot.node is not None:
                rclpy.spin_once(robot.node, timeout_sec=0.0)

            now = time.time()
            if now - last_t >= 1.0:
                dt = now - last_t
                fps = frame_cnt / dt if dt > 0 else 0.0
                print(f"[MainLoop] send_action rate ~ {fps:.1f} Hz")
                frame_cnt = 0
                last_t = now

            elapsed = time.time() - t0
            remain = period - elapsed
            if remain > 0:
                time.sleep(remain)
    except KeyboardInterrupt:
        print("\n[MainLoop] 用户中断，退出。")
    finally:
        mocap.disconnect()
        robot.disconnect()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


