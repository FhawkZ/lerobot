# FR3 RViz 场景与 mocap 控制

这个目录包含三部分：

- `start_fr3_rviz_scene.sh`：启动 FR3 的 RViz 可视化场景（`robot_state_publisher + rviz2`）。
- `fr3_rviz_arm_controller.py`：从 `mocap_leader.py` 迁移出的 **arm-only** 增量 IK 控制脚本，将 mocap 的 `RightHand` 位姿映射到 FR3 七关节，并发布到 `/joint_states`，用于驱动 RViz 里的 FR3 模型。
- `fr3_rviz_mocap_controller.py`：旧版本控制脚本（可保留做对比）。

## 1) 启动 RViz 的 FR3 场景

先准备终端 A：

```bash
cd /home/lqz/code/lerobot
source /opt/ros/humble/setup.bash
bash rviz/start_fr3_rviz_scene.sh
```

如果你想使用其他 URDF：

```bash
bash rviz/start_fr3_rviz_scene.sh /absolute/path/to/fr3.urdf
```

## 2) 启动 mocap -> FR3(RViz) 控制

终端 B（推荐新脚本）：

```bash
cd /home/lqz/code/lerobot
source /opt/ros/humble/setup.bash
python rviz/fr3_rviz_arm_controller.py \
  --mocap-udp-port 7012 \
  --mocap-poll-hz 120 \
  --control-hz 30
```

可选参数：

- `--urdf-path`：IK 使用的 URDF，默认 `src/lerobot/teleoperators/mocap_leader/fr3.urdf`
- `--ee-frame-name`：末端 link 名，默认 `fr3_link8`
- `--publish-topic`：关节发布话题，默认 `/joint_states`
- `--initial-joint-state-topic`：可选，用外部真实关节状态初始化 IK 当前位姿

---

## `mocap_to_linkerhand.py` 如何使用

这个脚本专门做 **手部**（LinkerHand L6）映射，与上面的 FR3 机械臂 RViz 控制是并行关系：

- 本目录的 `fr3_rviz_arm_controller.py`：只处理 arm（FR3 七轴）
- `mocap_to_linkerhand.py`：处理 hand（6 维手指开合/外展）

运行方式（终端 C）：

```bash
cd /home/lqz/code/lerobot
source /opt/ros/humble/setup.bash
python -m lerobot.teleoperators.mocap_leader.mocap_to_linkerhand
```

如果你要同时驱动“FR3(RViz) + LinkerHand(真实手)”：

1. 终端 A 启动 RViz 场景。
2. 终端 B 启动 `fr3_rviz_arm_controller.py`（arm）。
3. 终端 C 启动 `mocap_to_linkerhand.py`（hand）。

这样就对应了你现在的拆分目标：`mocap_leader.py` 的 arm 能力迁移到新脚本，hand 能力继续复用 `mocap_to_linkerhand.py`。
