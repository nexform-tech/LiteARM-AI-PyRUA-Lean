[English](litearm-real.md) | 简体中文

# LiteArm 真机硬件路径

本页说明 LiteArm / LiteGrip 真机工作站的适配：可单臂或双臂，每只 active 臂
可带夹爪或不带夹爪，相机可为头部/腕部中任意已配置子集。它与 RoboDojo 的
LiteArm 仿真适配相互独立：真机路径不导入、不启动 RoboDojo，不读取其任务、
布局、资产或 `.cache`，也不写 `eval_result`。

## 状态概览

| 项目 | 状态 |
| --- | --- |
| 离线端到端：runner → `robo` → toolkit → localhost RPC → 假设备 → 记录 | 已实现并测试 |
| adapter、配置/标定校验、FK/IK/TCP、规划、近似碰撞 | 已实现并离线测试 |
| RPC 去重、独占会话、取消、超时、断连、故障状态 | 已实现，并用真实 localhost socket 测试 |
| RGB-D / 多视角感知 | 已实现；用已知几何测试；SAM3 HTTP endpoint `http://sam3-host.example:8790` 已于 2026-11-05 验证 |
| 厂商 SDK 封装 | 在设备环境用 `litearm.testing.FakeTransport` 和 mock 夹爪测试 |
| 真机运动、急停保持/失能、固件抢占、标定精度 | **本次未在真机执行，需现场验收** |
| 真实相机采集（OpenCV / RealSense） | 已实现；本次未连接相机 |

## 深度与运动失败恢复

2026-11-10 的 `live-openai-full-094649` episode 暴露了罐体轮廓边界的
对齐深度污染，以及腕部限位附近的 IK 求解器发散。保存的第 35 步现在重放得到
罐体顶面 0.1801 m（高度 0.0513 m），原结果顶面为 0.3821 m。这些是保存图像
上的感知估计，并非独立实物测量。

- `locate` 现在拟合实测顶面，并用顶面的深度点计算中心和轮廓。
  RGB 轮廓投影继续作为独立估计。`measurement_quality` 报告各分量的可靠性和前景剔除情况。
- `table_projection` 提供配置台面上的无深度 x/y 和 yaw，物体高度测量失败时仍可
  返回。检查 `within_table_bounds`；透视可能使高处物体的台面投影落在桌外。
  不要把这个投影当作物体高度来下降。
- 显式台面查询返回 `source="calibrated_table"` 和配置中的支撑面 z。
  缺深度的 `back_project` 报错，不再跨视角匹配泛化物体或虚构台面/顶面交点。
- 空夹爪顶部抓取显式设置 `pitch=90`、`yaw=sight.grasp_yaw`、`roll=0`。
  省略角度会保留实测姿态。必要时先抬升，再在空旷区域显式使用 `path="auto"`
  调姿。`path="line"` 拒绝后不会自动执行关节摆动。
- IK 在限位附近使用受限步长和残差下降条件。关节、碰撞和接受容差继续生效。
  保存的 8 cm 抬升请求现在产生通过检查的直线路径，并报告 1.9 cm / 7° 的
  近似误差。这次重放执行了零个设备动作，不构成实物抓取验证。

在 `pyrua-lean` 使用缓存的 SAM3 掩码复现：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/miniconda3/envs/pyrualean-hw-agent/bin/python \
  -m pytest -q tests/test_litearm_real*.py
~/miniconda3/envs/pyrualean-hw-agent/bin/python tools/replay_episode_geometry.py \
  runs_hw/single-right-safety-test/live-openai-full-094649 \
  --config configs/litearm_real.json --step 35 \
  --mask-dir runs_hw/single-right-safety-test/diagnosis-20261110 \
  --prompt 'the red can' --prompt 'the white tabletop' \
  --output runs_hw/single-right-safety-test/diagnosis-20261110/replay.json
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避开宿主 Python 路径中无关的 ROS
pytest 插件。重放工具读取保存文件，完全不连接硬件服务。

2026-11-10 验证结果：116 项测试通过，1 项跳过（硬件测试及 prompt/robot/CLI
测试）；Ruff 通过。服务重启后的新只读快照得到罐体高度 0.0537 m、yaw 9.3°。
机械臂保持使能且无故障；关闭会话后服务为 READY，零个运动命令。证据位于
`runs_hw/single-right-safety-test/diagnosis-20261110/live-readonly/`。

新快照还暴露了 J6 读数比下限低 0.0000123 rad 的情况。FK 和轨迹实测起点
接受至多 0.001 rad 的限位读数噪声，同时保留实际测量值。IK 和下发轨迹点仍
受精确标定限位约束；超过观测容差的读数仍被拒绝。未修改标定限位或台面下限。

第二次 episode `live-openai-full-102149` 暴露了从这个实测起点细分时，小幅
J6 下限偏差被复制进 waypoint 0 的问题。命令插值现在把每个样本收回精确
关节限位内，同时保留实测 `q_start`。越过限位的 IK/home 终点仍被拒绝。
调整后的完整路径仍在 runner 上检查碰撞、步长、时长和速度，并由服务再次验证。

把保存的调用 19 显式设为 `path="joint"` 重放，会生成四个限位内的关节路点，
第一点 J6 精确等于
-1.5535470247268677 rad，并通过假设备 RPC 执行与回读测试。保存的沿用姿态
抬升至 z=0.29 请求仍不可达；此次修复没有放宽可达容差。失败引导也识别显式
复制状态角度的请求，因为它们保留了同一腕部姿态。空夹爪顶部抓取显式使用
`pitch=90`、`yaw=sight.grasp_yaw`、`roll=0`；仅在关节摆动有净空的区域使用
`path="auto"` 调姿。

本次修复后验证：121 项测试通过，1 项跳过；Ruff 通过。回归测试覆盖上下
限位、噪声起点细分、越界终点、保存的失败请求以及假设备 RPC 执行。证据位于
`runs_hw/single-right-safety-test/diagnosis-20261110/waypoint-fix/replay.json`。
未执行真机运动。新的 agent 进程会加载路点生成修复；无需放宽服务中的严格
路点检查。

## SDK 规划对照与低位抓取

2026-11-10 的核查将当前规划器与
`~/Code/NEXTFORM/litearm-python` 的 commit `5643e47` 对照。
其 `arm.py` 与设备环境安装的 SDK 逐字节一致。这些结果没有表明需要替换成
固件规划器。本地数值 IK 仍是独立求解器；本次核查不证明其数值结果与固件
IK 等同。

| 组件 | 结果 |
| --- | --- |
| 关节下发 | 每个路点使用 SDK `movej_sync`，同步进行关节插值 |
| 法兰与 TCP | SDK `get_tcp()` 返回法兰；标定偏移符合 `get_tool_tcp()` 语义，关节命令没有重复套用偏移 |
| 单位与姿态 | SDK 偏移使用米/弧度和 ZYX 旋转；`robo` 显式转换角度制及其 90° yaw 约定 |
| 官方 URDF | joint1–7 和固定 `ee_link` 的原点、轴与厂商 SE 压缩包完全一致 |
| 旋转插值 | 用最短弧、匀角速 SO(3) 插值替换矩阵混合；30 个随机姿态与 SDK 变换/插值函数的差异在浮点误差范围内 |
| 直线连续求解 | IK 失败或关节步长过大时细分粗采样；保留碰撞、限位、步长、时长和速度检查 |

`live-openai-full-103151` 第 44 步保存的预抓取姿态使用 2 cm TCP 采样间隔
和 0.05 rad 关节步长限制。这个间隔下的关节变化过大被报告为腕部奇异。
更细的连续求解能找到有效分支。规划器现在将失败的前进间隔减半，最多尝试
八次，不再把粗采样当作奇异性的证明。无法求解并通过验证的直线仍被拒绝。
`path="line"` 不执行关节回退；`path="auto"` 可以选择通过检查的关节路径。

`pitch=90` 时，当前工具体胶囊要求桌面上方 TCP z ≥ 0.1658 m：台面 z
0.1288 m + 边距 0.002 m + 工具体半径 0.035 m。这是本地碰撞模型约束，
不是 SDK 限制，本次没有放宽。配置中的指尖偏移为 0.046 m；实物短指尖的
接触深度仍未验证。路径可行不证明抓取成功。

| 保存的目标 `[-0.036, 0.432, z]`，yaw 14.4° | 修复后重放 |
| --- | --- |
| z = 0.162 m | `line` 和 `auto` 都拒绝 `right_tool_body` 碰撞 |
| z = 0.168 / 0.170 / 0.171 m | 两种请求均找到通过验证的直线路径，含 8–10 个路点 |
| z = 0.196 m | 可行，但记录中的真机抓取在这个高度空夹 |

这些路径使用空夹爪绕指向轴旋转半圈的等价姿态。持物时仍禁用这种对称性。
SDK 对每段进行关节空间插值，所以通过检查的直线是笛卡尔近似，而非固件
笛卡尔命令。

移动前使用只读 `robo.check_move()` 比较邻近高度：

```python
checks = [
    robo.check_move([-0.036, 0.432, z], pitch=90, yaw=14.4, roll=0, path="auto")
    for z in (0.171, 0.170, 0.168)
]
```

它返回可行性、选用路径、规划位置、路点数、时长、说明和工具体台面下限，
或附详细信息的错误。它读取新状态，执行零个设备动作。`move_to` 在执行前
重新规划。新 episode 使用场景对应的坐标和 yaw。
读取 `planned_xyz` 和 `note`；即使请求高度合适，接受的近似位姿仍可能太高，
无法接触物体。
**不要**在低目标因碰撞被拒后跳到高很多的抓取位置：在报告下限上方按几毫米
间隔比较高度。`line` 失败后，先对同一目标检查 `auto`，再决定下一次移动。
固定碰撞模型不包含可移动物体；执行关节绕行前检查它们的净空。

在 `pyrua-lean` 复现离线核查：

```bash
~/miniconda3/envs/litearm-hw-device/bin/python tools/audit_sdk_planning.py \
  --sdk-root ~/Code/NEXTFORM/litearm-python \
  --official-description ~/Code/NEXTFORM/assets/litearm/litearm_description_SE.zip \
  --config configs/litearm_real.json \
  --episode runs_hw/single-right-safety-test/live-openai-full-103151 --step 44 \
  --output runs_hw/single-right-safety-test/diagnosis-20261110/sdk-planning/audit.json
```

验证结果：agent 环境 126 项测试通过、1 项跳过；设备环境 4 项 SDK 测试通过。
单个历史固件法兰位姿与本地 FK 的误差为 3.04 µm / 0.000284°。这是一个
保存的姿态，不是新的固件 IK 或真机运动对照。核查执行零个设备动作。新的
agent 进程加载规划与指引改动；服务的轨迹验证仍保持严格。
最后的只读服务检查返回 READY，配置哈希匹配，无活动会话、执行中的命令或
故障。证据保存于
`runs_hw/single-right-safety-test/diagnosis-20261110/sdk-planning/service-status.json`。

## TCP 台面配置

针对操作者给出的约 12.5 cm 操作台，使用独立工作站配置
`configs/litearm_real.json`。配套标定文件为
`configs/calibration/litearm_real.json`。该配置保留
实测基座、TCP、相机变换及关节/速度设置。台面高度由操作者提供，本次没有
重新测量。

| 参数 | 数值与作用 |
| --- | --- |
| `table.z_m` | `0.125` m，机械臂基座世界坐标系中的台面高度 |
| `table.z_floor_m` | `0.125` m，此模式下整个工作空间的 TCP 最低高度 |
| `arms.right.tool_collision.table_check_mode` | `"tcp_plane"`，台面接触检查跳过工具体/指尖几何 |
| `arms.right.tool_collision.table_margin_m` | `0.002` m，桌面上方 TCP z ≥ `0.127` m；支持 `0.0`，仅检查平面而不加额外边距 |
| `arms.right.tool_collision.body_radius_m` | 保留 `0.035` m 用于自身/障碍物/另一臂检查；此模式下不再抬高 TCP 台面下限 |

机械臂连杆与台面的碰撞、关节限位、自身/另一臂碰撞和速度检查仍有效。
该配置用 TCP 检查台面与下限，所以不验证指尖与桌面的净空。省略
`table_check_mode` 时默认为 `geometry`，保留原有胶囊/指尖检查。
`table_margin_m` 作用于末端检查；机械臂连杆仍保留 2 mm 台面边距。

第 44 步离线重放允许 z = 0.130 / 0.150 / 0.162 / 0.170 m，拒绝
z = 0.125 / 0.126 m。未执行真机运动。这个独立配置尚未在当前服务启用。
服务和 agent 使用同一个工作站文件启动，才能通过配置哈希握手。

任务卡现在报告配置中的台面 z、世界坐标系、末端台面模式，并正确提取实测
TCP 变换中的平移。它不再把台面固定写成 z = 0，也不再把旋转元素写成
TCP 位置。

此次改动后验证：128 项测试通过，1 项跳过；Ruff 通过。

## 指尖 TCP 与实测到位

CAD 资产生成脚本 `~/Code/PyRUA/tools/litearm/build_litearm_assets.py`
定义指垫中点为法兰 +z = 0.1306 m，网格指尖为
0.0641 + 0.1125 = 0.1766 m。指尖在指垫中点前方 46 mm。
这些是 CAD 尺寸；当前实物短指尖仍未验证。如果旧 TCP z = 0.1575 m 的
疑似停短确实是触桌，会推算出约 29 mm 的延伸量，但单凭跟随误差不能确认
触桌，也不能测出指尖长度。

`tools/retarget_litearm_tcp.py` 通过局部 +x 平移组合新的法兰到 TCP 变换，
并同步换算腕相机外参和工具碰撞包络。`body_end_offset_m` 保留原工具体
端点，而不是把胶囊延伸到新指尖控制点。FK、SDK 偏移初始化、状态、规划和
任务卡使用生成的 TCP。

在 `pyrua-lean` 复现 CAD 候选；若所列输出已存在，使用
新的输出文件名：

```bash
~/miniconda3/envs/pyrualean-hw-agent/bin/python tools/retarget_litearm_tcp.py \
  --config configs/litearm_real.json \
  --offset-mm 46 --table-z-m 0.1288 --table-margin-mm 3 \
  --output-config configs/litearm_real.candidate.json
```

候选为只读、`executable=false`。指尖 TCP 台面最低高度为 0.1318 m；
全局 TCP 下限为 0.1288 m。仅在取得当前 TCP 到指尖的实物测量后，用
`--measured` 生成可用于 live 的配置。提供的 130.6 mm 指垫中点设置没有
补齐这个尺寸。配置换算和测试都不移动硬件。

SDK `Arm.q_tol` 默认 0.03 rad，工作站要求 0.01 rad，所以 SDK 可能返回
时服务仍判定未到位。封装现在将 SDK `q_tol` 设为
`execution.settle_tolerance_rad`，段结果也使用同一下限。指尖候选配置
使用 0.005 rad。这改变 SDK 报告到位的时机，不改电机增益，也不取消跟随
误差、超时或故障检查。真实接触仍会停止。真机改善效果要重试后才能验证。

使用指尖坐标时，目标应高于台面、低于物体顶面；不要给世界目标再加旧的
46 mm 偏移。`place_on` 需要实际配置 TCP 到持物底面的距离。

验证：agent 环境 129 项通过、1 项跳过；设备环境使用已安装 SDK 的 5 项测试
通过；Ruff 通过。TCP 迁移前后，相机世界变换和原碰撞包络保持不变。
指尖目标 0.140、0.133 m 通过离线检查；0.130 m 被台面下限拒绝。证据：
`runs_hw/single-right-table-plane/diagnosis-20261110/fingertip-tcp-audit.json`。
当前服务尚未以这些改动重启。

## 运行架构

两个进程：

1. **Runner / agent**（`pyrualean-hw-agent` 环境）运行模型生成的 Python，
   通过正常 `robo` API 调用 `LitearmRealRobot` → `HardwareToolkit`；
   toolkit 负责规划、感知、记录证据，并作为 RPC 客户端。
2. **硬件服务**（`litearm-hw-device` 环境）独占已配置的机械臂、LiteGrip
   和相机。它再次校验轨迹，执行同步关节段，采集实测状态，并管理会话、
   停止和故障状态机。

服务默认只监听 `127.0.0.1`。协议为 TCP 上的逐行 JSON；首版没有 TLS 和认证，
若跨机部署必须用防火墙或安全隧道保护。一个 runner 会话独占设备。

```
策略代码 -> LitearmRealRobot -> HardwareToolkit -> HardwareClient
                                                    |
                                        localhost TCP JSON
                                                    |
HardwareDaemon -> 规划复检 -> LiteArmStation -> LiteArm SDK
                                             -> LiteGrip SDK
                                             -> CameraManager
```

所有硬件 SDK、相机 SDK 只在设备创建函数内部导入。导入
`pyrualean.hardware` 或 `pyrualean.litearm_real` 不会连接设备、启动线程、
加载 GPU 模型或要求安装硬件 SDK。

## 已核对的 SDK 接口和版本

- `litearm-python` **2.1.0**，本地 `~/Code/NEXTFORM/litearm-python`，commit
  `5643e47`（git describe `v1.3.0-3-g5643e47`；包内 `__version__` 为 2.1.0）。
  它不是 PyPI 上的 1.3.0。
- `litegrip-python` **2.2.0**，本地 `~/Code/NEXTFORM/litegrip-python`，
  commit `6b5fdce`（git describe `v1.2.1-2-g6b5fdce`；源码树使用
  semantic-release 占位版本 `0.0.0`，已核对的安装报告为 2.2.0）。它不是
  PyPI 上的 1.2.1。

LiteArm 使用：

- `Arm.connect()`、`get_state(refresh=True)`（`Msg.value`）、`get_tcp()`、
  `get_tool_tcp()`、`set_tcp_offset(xyz, rpy)`；
- `movej_sync(q, speed)` 阻塞执行，`speed` 为 0..1；到位由实测状态判断。
  不用 `movej` 做线段，因为各轴独立 S 曲线使中间末端轨迹不可预测。
- `emergency_stop()` 是固件命令；`disable()`、`park()` 是另外的动作，不能
  当作受控停止。
- 不暴露 `move_js`：它需要 ≥10 Hz 重发和 0.1 s 看门狗；首版没有固定频率
  伺服调度、保活和停止机制。
- `get_tcp()` 返回法兰位姿；`get_tool_tcp()` 套用宿主侧 TCP 偏移。真机路径
  用实测关节 + 标定计算 TCP，并把固件法兰位姿作为交叉核对。

LiteGrip 使用：

- `LiteGrip(channel, can_id)`、`connect()`、`load_calibration(path=...)`、
  `get_state(wait=False)` 的 `position_mm`、`force_n`、`fresh`、
  `data_age_s`、`error_code`；
- `grasp(force_n, hold_s=0)` 会阻塞保持，由另一线程 `release()` 结束；
  每个夹爪只有一个持久 grasp 线程，重复 KEEP/close 不会重启夹取；
- 开启/部分开启用 `move_at_speed(target_mm, speed_mm_s)`；急停路径用
  `request_release(abort=True)`；`stop()` 只发一帧零力矩，不是安全级停止。

SDK 使用固定版本 wheel，不使用 editable 安装。用
`scripts/build_litearm_device_wheels.sh` 构建；LiteGrip 仓库的
semantic-release 占位版本在临时构建副本里替换成已核对的发布版本，不修改
输入 checkout。

## 安装

### Agent 环境

```bash
conda env create -f environments/litearm-agent.yml
conda activate pyrualean-hw-agent
python -m pip install -e .
```

Agent 环境不装硬件 SDK。SAM3 部署在 Jetson Thor，agent 只需要标准库 HTTP
客户端和站点文件中的 endpoint；离线测试用脚本化/HTTP segmenter，不依赖
`torch` 或 `sam3` 包。

### Jetson Thor 上的 SAM3

SAM3 作为 HTTP 服务部署在 Jetson Thor。在 Thor 上（`sam3` 包、torch 和
checkpoint 所在的机器）：

```bash
python src/pyrualean/hosts/sam3_http.py \
  --host 0.0.0.0 --port 8790 \
  --checkpoint ~/PyRUA/PyRUA-Lean/integrations/xpolicylab/PyRUA_Lean/checkpoints/sam3.pt
```

在 agent 机器上检查（设备网是直连，绕过 HTTP 代理）：

```bash
curl --noproxy '*' http://sam3-host.example:8790/health
# {"ok": true, "served": 0}
```

Agent 的站点文件指向它：

```json
"perception": {
  "sam3": {
    "mode": "http",
    "endpoint": "http://sam3-host.example:8790",
    "timeout_s": 60.0
  }
}
```

`sam3-host.example` 代表你网络中的 SAM3 服务地址；地址变化时改这里。
2026-11-05 已用 `/health` 和一次点提示分割验证该服务可连通。本版服务没有认证，
请限制在可信机器人网络或隧道内。`mode="local"` + checkpoint 在本地 GPU 机器上
仍然可用，但默认真机示例使用 Thor endpoint。

### 设备环境

```bash
conda env create -f environments/litearm-device.yml
conda activate litearm-hw-device
LITEARM_SDK_SRC=/path/to/litearm-python \
LITEGRIP_SDK_SRC=~/Code/NEXTFORM/litegrip-python \
scripts/build_litearm_device_wheels.sh
python -m pip install --no-index --find-links wheelhouse \
    litearm-python==2.1.0 litegrip-python==2.2.0
python -m pip install --no-deps -e .
```

驱动真机的机器不要用厂商 SDK 的 editable 安装。

设备环境已经固定 `pyrealsense2`（Python wrapper，wheel 内含 librealsense）和
OpenCV。Intel `realsense-viewer` **不是**运行时依赖；只作为可选诊断工具，启动
service 前必须关闭，因为它会独占相机。Linux 上还要安装 librealsense 的 udev
规则；pip wheel 不一定自动安装。

## 站点配置与标定

从 `configs/litearm_real.example.json` 复制出本机站点文件，从
`configs/calibration/litearm_real.example.json` 复制出标定文件。配置文件中的
相对路径以配置文件所在目录为基准。真实站点文件、标定、凭据和结果均被
`.gitignore` 忽略，示例保留。

关键配置：

- `service`：地址、端口、连接/请求超时、心跳周期/超时、最大时钟偏差。
  默认 localhost。
- `execution`：`mode`（`read-only`、`offline-fake`、`live`）、可选
  `active_arms`（默认 `["left", "right"]`；只装一只臂时设为 `["left"]` 或
  `["right"]`，此时另一只臂的 `devices` 条目和腕部相机可以省略，服务不会
  打开或控制它）、速度比例、
  配置的关节/TCP 上限、段长、稳定容差和命令超时。SDK 只暴露速度比例；
  加速度由固件负责，本层**不声称**已强制动态加速度限制。
- `devices`：每臂串口（优先用稳定的
  `/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_*` 符号链接，
  不要依赖会变化的 `/dev/ttyACM*`）和最低固件、每夹爪 CAN 通道/ID、可选
  标定文件、夹持力。
  该 active 臂没有 LiteGrip 时，`gripper` 可为 `null` 或整个省略：服务只
  打开、读取和停止机械臂，`state()` 的 `gripper`/`width_m` 为 `None`，
  OPEN/CLOSE 和 `place_on` 会明确拒绝，绝不会伪造夹爪。
  省略 `calibration_path` 时使用 SDK 随包出厂标定；`can1` 夹爪需要自己的
  文件，因为出厂文件声明的是 `can0`。
- `cameras`：驱动（`opencv`、`realsense`；`fake` 仅允许 offline-fake）、
  索引/序列号、分辨率、帧率、是否深度。`opencv` 还接受 `device`（显式路径，
  如 `/dev/video4`，用 `cv2.CAP_V4L2` 打开，优先于 `index`）和 `fourcc`
  （如 `"YUYV"`，在分辨率/帧率协商前设置）。整数索引无法寻址所有 V4L2
  节点；`device`/`fourcc` 对非 opencv 驱动会报错，opencv 相机必须有索引
  或设备路径。当前 D435/D435i 工位推荐：头部相机接 USB3，用
  `driver="realsense"` + 序列号 + `depth=true`；腕部相机接 USB2，用
  `driver="opencv"` + 稳定 `device` 路径 + `depth=false`。用 `lsusb -t`
  检查实际链路（5000M / 480M）；USB2 链路不足以承载头部彩色+深度。
  头部优先用 RealSense 序列号，腕部优先用 by-path/udev 符号链接，
  不要依赖会随重新插拔变化的 `/dev/videoN`。
- `perception`：SAM3 模式（Jetson Thor 用 `http`，也可用 `local`）、endpoint
  和超时、可选本地 checkpoint/GPU、体素、最大物体高度、最小有效深度比例，
  以及 `primary_camera`（默认 `"cam_head"`，必须是已配置相机）。runner 的
  `locate`、`segment`、`back_project`、`image`、`camera_meta` 在调用方省略
  `camera` 时使用它，因此只有腕部相机的站点也能直接 `locate(prompt)`。
- `snapshot`：最大数据年龄、相机/关节同步限制、刷新间隔、历史数。
- `results`：结果目录、设备动作预算、墙钟预算。
- `safety`：停止超时、静止确认阈值、home 抬升高度、故障后操作员复位策略。
- `fake`：仅 offline-fake 使用的确定性参数；live 忽略。

`execution.mode=live` 只有在所选机械臂和相机的标定 `"executable": true`
且质量状态不是示例/未验证时才允许。示例标定故意为 `executable: false`。

### 标定内容

- 世界原点：靠近机器人一侧桌沿中心；+x 向右、+y 向桌内、+z 向上，桌面
  z=0。
- 每臂完整基座位姿、法兰到 TCP 变换、关节映射（模型关节、SDK 索引、符号、
  零位偏移）、实测关节限位、安全 home。
- LiteGrip 实测闭合/张开宽度（归一化开口 =
  `(width - closed)/(open - closed)`）、夹持力、最小夹持宽度。
- 每个相机的 camera-to-world（头部）或 camera-to-tool（腕部）变换、内参、
  畸变、分辨率、是否有深度。
- 桌面范围和固定障碍。
- 来源、时间、版本和质量。

所有矩阵校验形状、有限值、旋转正交且行列式 +1。关节映射必须把模型关节
0..6 和 SDK 索引 0..6 各用一次，符号 ±1。限位必须落在打包 URDF 内。安全
home 必须在限位内。

标定必须**恰好**描述站点实际设备：`arms` 的键集合必须等于
`execution.active_arms`，`cameras` 的键集合必须等于配置的 `cameras`。
单臂/仅腕部相机的站点因此使用只含实际条目的裁剪标定文件；加载器既不要求
也不伪造缺失设备的条目。随包的两臂/三相机示例对默认配置仍然有效。
`load_calibration` 也接受显式 `arms=`/`cameras=` 选择器（默认全部），并执行
同样的精确集合校验。

### 人工测量步骤

`tools/litearm_real/calibrate.py` 只校验并写入测量文件，**不会**移动机械臂；
传 `--allow-motion` 会明确拒绝，而不是偷偷做运动采集。每项应人工测量并
记录方法：

1. **基座**：在已知关节构型下测基座原点/轴向，写 base-to-world 4x4。用多个
   位姿比较 `get_tcp()` 与 FK；观测构建器对超过 20 mm 的偏差报错。
2. **TCP**：测量抓取点相对法兰的位置/姿态（含 LiteGrip 安装），写
   `flange_to_tcp`。在 SDK 设同一偏移，用多个位姿比较 `get_tool_tcp()`。
3. **关节**：逐轴核对 SDK 顺序、方向、零位；手动或操作员低速工具逐轴运动，
   记录映射和实际软限位。不要猜符号。
4. **夹爪**：LiteGrip SDK 自带 `src/litegrip/factory_calibration.json`
   （`calibrated: true`、`channel: can0`），所以 `can0` 夹爪不需要额外文件，
   不带参数的 `load_calibration()` 会用它。SDK 故意不会在 `can1` 上隐式复用
   这份文件；请对该夹爪运行厂商的 `zero()`/`calibrate_guided()`/
   `calibrate_manual()`，并把 `devices.<arm>.gripper.calibration_path` 指向
   生成的 `~/.litegrip/<channel>_calibration.json`。用卡尺测闭合/全开宽度，
   验证命令停止在标定行程内。力是 SDK 名义换算，不是标定力，不能用于安全
   决策。下面的 `calibrate.py` 只写**站点**标定（base/TCP/相机），不生成
   这个厂商夹爪文件。
5. **相机**：用标定板测内参/畸变；头部测 camera-to-world，腕部测
   camera-to-tool。分辨率必须一致。RealSense 驱动会拒绝与标定差异超过 2%
   （主点 4 px）的设备内参。RealSense 彩色流请求 `rs.format.bgr8`，返回前
   转为 RGB：D435i 固件 5.17.0.9 在 640x480@30 下 `rgb8` 彩色流会超时，
   `bgr8` 可稳定同时输出彩色和深度。若 OpenCV 相机无法用整数索引打开
   V4L2 节点，设置 `cameras.<name>.device` 为路径（如 `/dev/video4`），
   必要时设 `fourcc`（如 `"YUYV"`）；驱动用 `cv2.CAP_V4L2` 打开路径，并在
   宽/高/帧率之前设置 FourCC。
6. **Home**：选一个无碰撞关节构型作为 `safe_home_rad`，用规划器验证抬升和
   返回路径。它不是 SDK 零位。

把工作副本中的示例值全部替换为实测值，把 `source` 改为真实测量记录，
把 `quality.status` 改为已测量状态，然后写入：

```bash
python tools/litearm_real/calibrate.py \
  --from /abs/path/to/measured_calibration.json \
  --out configs/calibration/litearm_real.json \
  --measurement-quality measured \
  --mark-executable
```

输入文件仍带 `example`、`template` 或 `unverified` 标记时，工具拒绝标记为
可执行。只有全部测量并复核后才设 `executable=true`。未验证/示例标定会被
live 拒绝。

## 启动服务

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_MODE=live \
LITEARM_HW_LOG=/abs/path/service.log \
scripts/litearm_hardware_service.sh
```

离线验证用 offline 配置和 `LITEARM_HW_MODE=offline-fake`。服务：

- 不自动 enable、reset、清故障或回零；
- live 下若臂或夹爪未使能/有故障则拒绝启动（由操作员检查后在服务外处理）；
- 同一时刻只允许一个 runner 会话；
- read-only 允许快照，但拒绝运动。

## 运行 episode

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_INSTRUCTION='把桌上的红色积木放到托盘里' \
LITEARM_HW_MODE=live \
LITEARM_HW_BUDGET_S=1200 \
scripts/litearm_hardware_play.sh
```

等价直接命令：

```bash
pyrualean play \
  --robot litearm_real \
  --arm cells \
  --primitives no-vla \
  --hardware-config /abs/path/station.json \
  --task pick_place \
  --instruction "把桌上的红色积木放到托盘里" \
  --out /abs/path/results_hw/run_id
```

只有显式 `--robot litearm_real` 才选择真机 adapter；没有环境变量会偷偷改变
默认 litearm 仿真后端。`read-only` 仍连接并感知，但拒绝运动；
`offline-fake` 使用合成设备和相机，绝不会因真实驱动失败而自动回退；
`live` 要求所选 active 臂和已配置相机的标定可执行且质量已测量。

## 停止语义

- **普通取消**：在同步段边界后用新鲜状态读回确认，臂保持实测位姿，服务进入
  `stopped`。之后再发轨迹必须重启服务。
- **超时/心跳丢失/显式急停**：服务调用固件急停并进入 `fault`。结果记录停止
  是否得到确认；未确认时报告 `stop_unconfirmed`，不能声称停止成功。
- 服务不把 `disable()` 或 `park()` 当作受控停止。
- 故障不会自动清除；重连不会恢复运动。
- 急停在真机上的保持/失能行为、能否抢占阻塞段，均为**现场验收项**。

## 结果与诊断

episode 目录包含：

- `prompt.txt`、`calls.jsonl`、`tool_turns.json`、`result.json`；
- `steps/NNNNNN/`：`snapshot.json`、`observation.json`、`camera_meta.json`、
  `device.json`、`artifacts/...`（PNG/NumPy）；
- `episode.json`：停止证据和成功语义；
- `hardware_host.json`、`card/`（图片和 `card.json`）。

`State.step` 是观测快照编号；`steps_used`/`env_steps` 是已执行轨迹段或夹爪
动作数。`environment_success` 为 false，`environment_success_known` 为 false，
`task_success` 为 `not_measured`。模型 `finish(status="success")` 单独记录，
不是任务成功判定。

## 诊断

- `tools/litearm_real/cameras.py`：不依赖 Viewer 的相机检查，枚举 RealSense
  序列号和 USB 链路，采集头部彩色+深度、腕部彩色，写 JSON/PNG。头部序列号
  必须协商到 USB3：
  ```bash
  python tools/litearm_real/cameras.py \
    --head-serial HEAD-SERIAL-EXAMPLE \
    --wrist-device /dev/v4l/by-path/... \
    --out runs_hw/camera-check
  ```
- `tools/litearm_real/smoke.py`：只读检查握手、哈希、时钟、快照、TCP 和停止
  证据。`--allow-motion` 可做小运动/单物体抓放；本次未在真机执行该路径。
- 服务日志记录状态读失败、故障原因和停止确认。
- 检查 `steps/*/camera_meta.json` 与 `device.json` 的一致性。
- `step=-1` 的 `camera_meta` 与 `image` 来自同一固定快照；之后刷新状态会创建
  新快照。

## 测试

Agent 环境（无硬件 SDK、无本地 SAM3/torch）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_litearm.py tests/test_litearm_real.py \
  tests/test_litearm_real_config.py tests/test_litearm_real_geometry.py \
  tests/test_litearm_real_perception.py tests/test_litearm_real_execution.py \
  tests/test_litearm_real_cameras.py tests/test_cli.py -q
```

设备环境：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_litearm_real_devices.py -q
```

设备测试使用 `litearm.testing.FakeTransport` 和 mock LiteGrip，不是物理验证。
执行测试使用真实 localhost TCP 和合成站点，不是只 mock 客户端方法。

## 已知限制 / 待现场验证

- 所有真实运动、夹持力、急停、固件行为、线缆时序和标定精度都待现场验证。
- 碰撞几何是保守的胶囊/盒体近似，不是 mesh 精确碰撞，也不构成避障认证。
- SDK 只暴露速度比例；段长和超时是运行护栏，不是动态极限证明。
- 首版服务没有 TLS/认证；请限制在本机或受保护网络。
- SAM3 预期部署在 Jetson Thor；示例指向上面已验证的地址，但 HTTP 服务必须
  正在运行。不可达时感知调用明确失败；本地 `sam3`/checkpoint 仍作为可选后备。
- RealSense 需要 `pyrealsense2` 和匹配固件；本次未连接相机。
- 示例配置/标定不可执行；未测量前不要设 `executable=true`。

## 正方体抓取开发原语（禁用）

`robo.grasp_cube` 完成一次对中、垂直夹持的正方体抓取。
它**默认禁用**，需要站点显式设置 `manipulation.enable_cube_grasp=true`。
这一版只做离线测试，活动站点配置和服务均未修改。
只有操作者批准这一版后，才能启用并进行真机试验。

输入应为新测量的 `center_xy`、物体 `top_z`、局部支撑面 `support_top`、
闭合方向截面尺寸 `side_m` 和闭合轴角度 `yaw`（`locate(...).grasp_yaw`）。
还需提供 `pad_height_m`，即从指尖向上量取、已核验的有效接触长度。
碰撞胶囊尺寸不能代替指垫测量。原语要求 TCP 为 `fingertip_midpoint`，
且额外指尖偏移为零。

流程为到上方、垂直下降、闭合、等待保力稳定、建立稳定基准，最后垂直抬升 15 mm。
默认夹持测量高度的 65%，另留 1 mm 到达余量：
`tip_z = top_z - 0.65 * (top_z - support_top) - 0.001`。
闭合前核对实际水平中心误差不超过 2 mm、垂直和指面方向误差不超过 3°，
以及实际几何接触高度超过物体的一半。两指尖都必须离支撑面至少 3 mm。
指垫高度、台面边界、夹爪行程、工具/地板约束不满足，或实际到达过浅、未对齐时，
原语拒绝抓取。

CLOSE 后至少等待 1 s，并要求新鲜的开度和名义保力反馈持续稳定至少 0.5 s。
开口宽度与输入截面尺寸的差不超过 3 mm，稳定窗口内变化不超过 0.5 mm。
报告的名义保力须达到配置值的 95% 并稳定；它不代表标定后的接触力。
满足这些条件后，才把稳定开度保存为现有滑落保护的搬运基准。
验证抬升失败时保留夹持以供恢复；抬升前失败且服务正常时，在支撑面上松开。
原语不自动重试或回零。

`RealGrasp` 返回 `grasped`、`failure_stage`、实际 `overlap_fraction`、
`baseline_width_m`、`baseline_opening`、`lift_reached`、`slip`、
`tcp_to_bottom`、动作数量和逐阶段证据。
`grasped` 表示几何和持续夹持验证通过，不是独立的物体身份或任务成功判定。
输入几何、实物 TCP 和指垫尺寸仍需核验。

从 `pyrua-lean` 运行离线测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/miniconda3/envs/pyrualean/bin/python \
  -m pytest -q tests/test_litearm_real_grasp.py
```

仅临时假设备测试配置开启该原语；测试不连接当前真机服务。

## PARK 相机统一与固定叠放（仅供审核）

操作者已在 2026-11-11 批准启用下一轮试用。下文的审核配置仍保持只读；
当前选定的启用配置和启动流程保留在本地站点配置中。

新审核配置的腕相机标定来自本地的 PARK 眼在手上导出；按发布示例创建
`configs/calibration/litearm_real.json`。
导入器按 `inv(flange_to_tcp) @ T_flange_cam @ diag(1,-1,-1,1)` 重建腕相机外参。
再次导入仍从 PARK 重建，不会对已导入矩阵重复翻转。
运行时相机元数据与解析后的标定统一使用 `camera_frame="opengl"`
（+x 向右、+y 向上、沿 −z 观察）。显式 OpenCV 输入只在边界转换一次；
未知坐标标签会被拒绝。深度反投影、RGB 投影与射线/平面交点共用
`hardware/camera_geometry.py`，策略代码不得另试一种相机坐标约定。

腕相机内参也来自 PARK 导出。1280×720 内参通过高度等比缩放、水平居中裁剪映射到
640×480，映射记录在 `handeye_source` 中。
结果与先前 640×480 内参在舍入精度内一致，但实物流的裁剪方式和标定精度仍需现场核验。

已准备的文件：

- 站点：`configs/litearm_real.json`。
- 标定：`configs/calibration/litearm_real.json`。
- 两个正方体原语均关闭，标定为 `executable=false`，执行模式为 `read-only`。
  未替换活动服务或活动配置。

操作者后续开启两个正方体原语后，
`robo.stack_cube(source_prompt, support_prompt, pad_height_m=...)` 执行完整流程：

1. 任何运动之前，对两块各测量两次。三维测量失败、不一致或物体超出台面边界时拒绝。
2. 冻结支撑块中心与顶面；抓取源方块并保存实际 TCP 位置和稳定夹持基准。
   固定本次抓取的水平物体偏移及 TCP 到物体底面的高度。
3. 垂直抬升，使被抓物体底面至少高于较高物体 30 mm。保持夹持姿态，随后在此高度
   一次水平移动到 `support_center_xy - held_offset_xy`。
4. 核对实际 TCP 和夹持状态，并对两块各做一次验证测量。
   支撑块与被抓方块中心须在 3 mm 内对齐，高度须在 3 mm 内一致。
   测量缺失或矛盾就停止，不修改落点，也不做探索式纠偏。
5. 使用 `place_on` 下降、松开、抬升和撤离。张开未稳定或撤离失败时报告操作未完成。
   不自动回零。

`RealCubeStack` 返回冻结的支撑块、夹持偏移、目标 TCP、搬运高度、对齐结果、
逐阶段证据及动作数量。`completed` 仅表示测量支持的执行完成，独立任务成功仍未知。

离线回归覆盖独立 OpenCV 投影/深度往返、实际 PARK 导入及重复导入、
矛盾或缺失的几何、禁用开关，以及 runner 到假设备 RPC 的完整叠放流程。
未执行真机动作，也未调整速度或优化航点。

对于腕相机坐标错误的旧记录，回放脚本可配合审核配置使用
`--use-calibrated-camera-poses`。它根据记录中的拍摄时刻关节与所选标定重建腕相机位姿，
明确标记输出为标定回放，设备动作数为零，不重写历史记录。

## 目标跟踪与顶面测量

真机 runner 在一次 episode 内跟踪每个文本目标，帮助操作员区分目标方块和
其他同色物体。下一轮的抓取和堆叠开关继续保持禁用。这些修改已有离线回归
覆盖；物理精度尚未验证。

抓取前，对同一目标在两次不同的采集中调用 `locate`，并要求
`geometry_confirmed=True`。重复的相机时间戳不计入确认。第一次几何测量形成
固定参考。参考相机中的无效观测会清除确认计数，但不替换参考。辅助相机缺少
RGB 三维几何不会清除参考相机已确认的深度；相互矛盾的实测三维结果仍会使
确认失效。默认位置容差为 8 mm，
顶面容差为 4 mm，尺寸容差取 4 mm 和参考尺寸的 30% 中的较大值。
这些数值是软件拒绝边界。

后续 `locate` 和 `segment` 会把 SAM3 输入裁剪到参考三维搜索区域的投影范围。
过大、截断或区域外的检测会被拒绝。即使调用者降低 `min_score`，配置的
SAM3 最低分数仍然有效。原始结果记录 `search_roi_rc` 和相机采集时间戳。
后续检查应使用相同的文本提示。

闭合稳定后，只有恰好一个附近的已确认参考与夹爪宽度一致时，才会关联夹持目标。
完整三维偏移和朝向通过实际测得的 TCP 姿态传播。这是搜索预测，不是物体几何
测量。抓取身份有歧义或宽度反馈变化时，返回 `target_unconfirmed`。
张开夹爪会清除夹持参考。

RGB-D 几何通过有界 RANSAC 拟合顶面，并剔除预期目标体积外的前景深度。
平面 RMS 上限为 2 mm；中心和轮廓确认要求至少 75% 的检测掩码具有符合拟合
竖直物体顶面和侧面几何的有效深度，其中顶面深度至少覆盖检测掩码的 20%。
这样可以支持斜视，而不会把正常侧面深度当作遮挡。
`measurement_quality` 分别报告 `height_reliable`、`center_reliable` 和
`footprint_reliable`。顶面可见而中心受遮挡时，只返回诊断性的
`partial_measurement`：主要几何不可用，且 `measurement_failed=True`。
无效观测返回 `error_code="target_unconfirmed"`。纯 RGB 视觉壳的可靠性根据
体素分辨率、视角视差、体积占据率、支撑面接触，以及拟合方块与两幅完整
轮廓的一致性计算，不再一律为 false。有效拟合可在两次采集后建立参考；
有歧义或不一致的拟合仍保持未确认。量化不确定性单独报告，不证明物理
标定精度。

悬空物体的高度来自抓取前的实测物体，不来自与桌面的距离。已跟踪的抓取会
自动选择 `geometry_mode="held"`。直接几何查询可提供
`geometry_mode="held", object_height_m=...` 和实测高度；对于其他已知支撑面
上的物体，可提供 `support_z_m=...`。查询仍需要实测深度或匹配的标定视角。
夹持物体偏离竖直超过 5° 时，水平顶面模型会返回未确认。
`rgb_footprint` 和 `table_projection` 是独立估计，不替换实测中心或失败的目标。

**不要**通过降低分数、更换提示或使用预测值绕过身份和几何检查。
原目标的两次新有效采集可以重新建立跟踪确认。

`live-stack-143454` 的离线重放接受干净绿块的快照 70 和 71，顶面 z 分别为
0.1349 和 0.1367 m。第二次采集获得确认；受污染的快照 145–147 被拒绝，
不会更新支撑块高度。被遮挡红块的重放仍返回未确认，不会选择红杯。
这些结果验证拒绝逻辑，不证明物理放置精度。

`live-stack-172530` 前记录的 HORAUD 头相机场景暴露了一个精修回归：
RANSAC 选择的平面满足 10° 斜率上限，但无约束最小二乘随后跑出上限，
导致有效深度被拒绝。绿块和黑块内部深度均为 100% 有效，残差约 1 mm。
现在精修在相同约束内求解，不再因精修跑出边界而拒绝原本有效的拟合。
保存的黑、绿块帧通过深度几何和两次采集确认；不一致的黑块 RGB 视觉壳
仍因完整轮廓不匹配而失败。紧凑深度/掩码测试数据保存在
`tests/data/park_horaud_cube_depth.npz`。

agent 指引也把失败的几何采集限制为最多两个新规划视角，然后停止。
头部几何已确认时，腕部缺深度不是继续移动的理由。这次修改不改变 SDK
到位阈值，也不启用任何方块原语。新启动的 agent 进程会加载修改。

协作式取消后，心跳会保留已停止会话，供读取指令结果和停止证据；后续动作
仍被拒绝。这消除了取消与回读之间的竞态，服务重启后加载该修复。

在 `pyrua-lean` 中，通过配置的 SAM3 服务重放保存的
RGB/深度。该命令不连接机械臂设备：

```bash
PYTHONPATH=src ~/miniconda3/envs/pyrualean-hw-agent/bin/python \
  tools/replay_target_perception.py --config configs/litearm_real.json \
  --episode runs_hw/single-right-park-unified/live-stack-143454 \
  --steps 70 71 145 146 147 --prompt 'the green block' \
  --output runs_hw/perception-green-replay.json
```

工具拒绝覆盖已有输出文件。`--hold-step 86 --held-arm right` 可从保存的
快照重建稳定夹持，用于检查后续帧。报告记录配置和感知源码哈希，以及零次
设备动作。

## API reference

物体测量与允许操作的台面范围分离。有效深度点不再因台面的 x/y 边界被丢弃；
多视角重建在相机轮廓射线内搜索，体素数量仍受限。
`locate` 返回测量坐标，同时单独提供 `within_table_bounds`、
`footprint_within_table_bounds` 和 `bounds_error`。
范围外高度仍以配置台面为参考，是对该平面的外推。
`stack_cube` 保留台面范围拒绝，在任何闭合或运动之前返回
`error_code="outside_calibrated_table"`、实测位置与配置边界。
感知成功不代表可以在未测量实际支撑面几何的情况下扩展操作区。

<!-- api-reference: litearm_real primitives=no-vla -->
