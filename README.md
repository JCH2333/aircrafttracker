# Aircraft Tracker

自动检测并追踪视频中的民航客机，以飞机为中心进行画面稳定，保持原始分辨率和色彩格式。

## 功能

- **遮挡鲁棒追踪**：SAM 2.1 视频分割锁定飞机身份，掩码内 LK 特征点与 RANSAC 更新固定机体参考点
- **状态机与双向补轨**：区分正常、遮挡和丢失状态，并通过 Kalman + RTS 使用遮挡前后观测平滑补全轨迹
- **候选门控**：结合预测位置、历史面积和长宽比拒绝大面积前景及错误目标
- **人工复核**：自动列出低置信片段，可逐帧拖框修正、撤销并只重算相邻可靠锚点之间的区间
- **兼容后端**：GPU/SAM 2 不可用时自动回退经过门控的轮廓模板追踪
- **逐帧居中**：飞机固定在画面中心，周边黑边原样保留
- **10-bit 色彩**：完整保留 yuv422p10le / yuv420p10le 色彩精度
- **原始分辨率**：支持 4K 和 6K，输出与输入分辨率一致
- **音频直通**：原始音频流无损拷贝到输出

## 系统要求

- Windows 10/11
- Python 3.12+
- NVIDIA GPU（推荐使用混合后端；CPU 自动使用 legacy）
- FFmpeg（需在 PATH 中可用）

## 安装

CPU / legacy：

```bash
git clone https://github.com/JCH2333/aircrafttracker.git
cd aircrafttracker
pip install -r requirements.txt
```

NVIDIA GPU / hybrid：

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

## 使用方法

### 双击启动

1. 确保已安装 Python 3.12+ 和依赖（`pip install -r requirements.txt`）
2. 双击 `启动GUI.bat`
3. 脚本会自动检查环境，然后打开图形界面

### GUI 模式

```bash
python -m stabilize.main --gui
```

- 单文件：选择输入视频 → 选择输出目录 → 点击"开始处理"
- 批量：勾选"批量模式" → 选择文件夹 → 勾选要处理的文件 → 开始处理
- 实时进度条显示 Pass 1（追踪）和 Pass 2（渲染）进度

### CLI 单文件

```bash
python -m stabilize.main 素材/P1021917.MOV -o 输出/result.MOV \
  --tracking-backend hybrid --review
```

使用已有人工锚点：

```bash
python -m stabilize.main 素材/P1021917.MOV \
  --track-file 输出/P1021917.track.json
```

强制 CPU 兼容后端：

```bash
python -m stabilize.main 素材/P1021917.MOV \
  --tracking-backend legacy
```

### CLI 批量

```bash
python -m stabilize.main 素材/ --output-dir 输出/
```

### 查看视频信息

```bash
python -m stabilize.main 素材/P1021917.MOV --info
```

### 完整参数

```
usage: python -m stabilize.main input [-h] [-o OUTPUT] [--output-dir DIR]
                                      [--detector {torchvision,yolo}]
                                      [--tracking-backend {hybrid,legacy}]
                                      [--sam2-model MODEL]
                                      [--track-file FILE] [--review]
                                      [--conf FLOAT] [--border {reflect,replicate}]
                                      [--crf INT] [--preset PRESET]
                                      [--gui] [--debug] [--info]

选项:
  --detector       检测后端 (默认: torchvision)
  --tracking-backend
                   hybrid: SAM 2 + 掩码特征；legacy: 轮廓模板
  --track-file     人工锚点和问题片段 sidecar
  --review         Pass 1 后暂停并复核低置信片段
  --conf           检测置信度阈值 (默认: 0.5)
  --crf            x264 质量, 越小越好 (默认: 18)
  --preset         编码速度 preset (默认: medium)
  --codec          编码器: libx264 (CPU) / h264_nvenc (NVIDIA GPU, 默认GUI开启)
  --border         边缘模式 (默认: constant / 黑边)
  --gui            启动图形界面
  --debug          调试模式
  --info           显示视频信息后退出
```

## 素材要求

最佳效果：
- 民航客机（刚体，形变极小）
- 飞机在画面中占比 > 5%
- 无明显运动模糊

已知限制：
- 超过 90 帧的无可靠观测区间应人工确认
- 飞机长期完全出画且没有后续可靠锚点时需要人工框选
- SAM 2 的 Windows 可选后处理扩展若未编译，会自动跳过孔洞填充，不影响主体传播
- CPU 编码速度较慢；GUI 默认开启 NVENC 硬件编码（NVIDIA GPU 需支持）

## 技术栈

- **检测**：PyTorch Faster R-CNN (COCO 预训练)
- **混合追踪**：SAM 2.1 + 掩码 LK 光流 + RANSAC 部分仿射
- **兼容追踪**：Sobel 梯度幅值 + NCC 模板匹配
- **轨迹**：二维常加速度 Kalman + RTS 反向平滑
- **I/O**：PyAV (FFmpeg)
- **GUI**：CustomTkinter

## 许可证

GNU General Public License v3
