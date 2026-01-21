# 异步/分布式推理快速指南

## 1. 角色与职责
- **PolicyServer（推理服务端）**：仅负责加载模型、预处理/后处理、推理；不需要具体机器人硬件配置。
- **RobotClient（机器人端）**：连接机器人硬件、采集观测、将观测发送到服务器、接收并执行动作。

## 2. 基本启动命令
在服务器上（默认端口 8080）：
```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080
```

在机器人端（示例）：
```bash
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=fr3_follower \
    --robot.id=<your_robot_id> \
    --policy_type=act \
    --pretrained_name_or_path=<model_path_or_hub_id> \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

## 3. 端口与网络
- `PolicyServer` 使用 gRPC，当前代码采用 `add_insecure_port`，无内置加密。
- 生产或跨网段建议用 SSH 隧道；内网且可信可直接暴露端口（自行加防火墙/ACL）。

## 4. SSH 隧道示例（推荐）
适用于无法开放新端口或需要加密传输的场景。示例服务器 SSH 信息：
`ssh -p 2222 root@172.30.4.178`

### 4.1 在服务器上启动 PolicyServer（仅监听本机）
```bash
ssh -p 2222 root@172.30.4.178
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080
```

### 4.2 在本地建立隧道
```bash
ssh -p 2222 -f -N -L 18080:127.0.0.1:8080 root@172.30.4.178
# 如果 18080 被占用，换一个本地空闲端口即可
```

### 4.3 本地测试隧道
```bash
curl -I http://127.0.0.1:18080/
# 看到 HTTP/1.0 200 OK 即隧道可用
```

### 4.4 通过隧道运行 RobotClient
```bash
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:18080 \
    --robot.type=fr3_follower \
    --robot.id=<your_robot_id> \
    --policy_type=act \
    --pretrained_name_or_path=<model_path_or_hub_id> \
    --policy_device=cuda \
    --actions_per_chunk=50
```

### 4.5 关闭隧道
```bash
pkill -f "ssh.*18080:127.0.0.1:8080"
```

## 5. 直接暴露端口的简要指引（如需）
```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
```
记得在防火墙放行 8080/TCP，并考虑额外的 TLS/认证层。

## 6. 常用可调参数
- `--actions_per_chunk`：每批动作数量，平衡带宽和延迟。
- `--chunk_size_threshold`（客户端）：队列剩余比例阈值，决定何时发送新观测。
- `--policy_device`：服务器推理设备，如 `cuda`、`cpu`、`mps`。
- `--fps`：期望控制频率，影响 `environment_dt`。

## 7. 快速排查
- 隧道通断：`curl -I http://127.0.0.1:<local_port>/`
- 端口占用：`lsof -i :18080`（本地）/`lsof -i :8080`（远端）
- 服务器在跑：`ps aux | grep policy_server`


