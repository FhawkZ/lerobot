#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
URDF_PATH="${1:-${REPO_ROOT}/src/lerobot/teleoperators/mocap_leader/fr3.urdf}"

if [[ ! -f "${URDF_PATH}" ]]; then
  echo "URDF not found: ${URDF_PATH}"
  exit 1
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "ROS_DISTRO not set. Please source your ROS2 environment first."
  echo "Example: source /opt/ros/humble/setup.bash"
  exit 1
fi

resolve_franka_description_share() {
  # Prefer workspace-installed package if available.
  if command -v ros2 >/dev/null 2>&1; then
    local pkg_prefix
    pkg_prefix="$(ros2 pkg prefix franka_description 2>/dev/null || true)"
    if [[ -n "${pkg_prefix}" && -d "${pkg_prefix}/share/franka_description" ]]; then
      echo "${pkg_prefix}/share/franka_description"
      return 0
    fi
  fi
  # Fallback to system install.
  if [[ -d "/opt/ros/${ROS_DISTRO}/share/franka_description" ]]; then
    echo "/opt/ros/${ROS_DISTRO}/share/franka_description"
    return 0
  fi
  return 1
}

FRANKA_DESCRIPTION_SHARE="$(resolve_franka_description_share || true)"
if [[ -z "${FRANKA_DESCRIPTION_SHARE}" ]]; then
  echo "Cannot locate franka_description package."
  echo "Please source a workspace/environment that provides franka_description."
  exit 1
fi

ROBOT_DESCRIPTION="$(
  sed "s|package://franka_description/|file://${FRANKA_DESCRIPTION_SHARE}/|g" "${URDF_PATH}"
)"

RSP_PARAMS_FILE="$(mktemp /tmp/fr3_rsp_params.XXXXXX.yaml)"
cleanup() {
  kill "${JSP_PID:-}" >/dev/null 2>&1 || true
  kill "${RSP_PID:-}" >/dev/null 2>&1 || true
  rm -f "${RSP_PARAMS_FILE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
  echo "/robot_state_publisher:"
  echo "  ros__parameters:"
  echo "    robot_description: |"
  # YAML block scalar content needs indentation.
  while IFS= read -r line; do
    echo "      ${line}"
  done <<< "${ROBOT_DESCRIPTION}"
  echo "/joint_state_publisher:"
  echo "  ros__parameters:"
  echo "    robot_description: |"
  while IFS= read -r line; do
    echo "      ${line}"
  done <<< "${ROBOT_DESCRIPTION}"
} > "${RSP_PARAMS_FILE}"

echo "[1/3] Starting robot_state_publisher..."
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args --params-file "${RSP_PARAMS_FILE}" &
RSP_PID=$!

sleep 0.5
echo "[2/3] Starting joint_state_publisher..."
ros2 run joint_state_publisher joint_state_publisher \
  --ros-args --params-file "${RSP_PARAMS_FILE}" &
JSP_PID=$!

sleep 0.5
echo "[3/3] Starting RViz2..."
rviz2
