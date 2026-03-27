#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    robot_ip = LaunchConfiguration("robot_ip")
    namespace = LaunchConfiguration("namespace")
    arm_id = LaunchConfiguration("arm_id")
    load_gripper = LaunchConfiguration("load_gripper")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")

    franka_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("franka_bringup"), "launch", "franka.launch.py"])
        ),
        launch_arguments={
            "arm_id": arm_id,
            "namespace": namespace,
            "robot_ip": robot_ip,
            "load_gripper": load_gripper,
            "use_fake_hardware": use_fake_hardware,
        }.items(),
    )

    # 等待底层 ros2_control 节点启动后再加载位置控制器
    spawn_joint_group_position = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                arguments=["joint_group_position_controller", "--controller-manager-timeout", "30"],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="172.16.0.1", description="FR3 机器人 IP"),
            DeclareLaunchArgument("namespace", default_value="NS_1", description="机器人命名空间"),
            DeclareLaunchArgument("arm_id", default_value="fr3", description="机械臂型号"),
            DeclareLaunchArgument("load_gripper", default_value="false", description="是否加载夹爪"),
            DeclareLaunchArgument(
                "use_fake_hardware", default_value="false", description="是否使用假硬件（仅调试）"
            ),
            franka_launch,
            spawn_joint_group_position,
        ]
    )
