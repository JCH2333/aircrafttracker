# Aircraft Tracker

自动检测并追踪视频中的民航客机，以飞机为中心进行画面稳定，保持原始分辨率和色彩格式。

## 功能

- **轮廓匹配追踪**：Sobel 梯度幅值 + NCC 模板匹配，对前景遮挡、模糊失焦、部分出画鲁棒
- **逐帧居中**：飞机固定在画面中心，周边黑边原样保留
- **10-bit 色彩**：完整保留 yuv422p10le / yuv420p10le 色彩精度
- **原始分辨率**：支持 4K 和 6K，输出与输入分辨率一致
- **音频直通**：原始音频流无损拷贝到输出

## 系统要求

- Windows 10/11
- Python 3.12+
- NVIDIA GPU（可选，CUDA 版更快）
- FFmpeg（需在 PATH 中可用）

## 安装

```bash
git clone https://github.com/JCH2333/aircrafttracker.git
cd aircrafttracker
pip install -r requirements.txt
```

## 使用方法

### 🖱️ 双击启动（推荐）

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
python -m stabilize.main 素材/P1021917.MOV -o 输出/result.MOV
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
                                      [--conf FLOAT] [--border {reflect,replicate}]
                                      [--crf INT] [--preset PRESET]
                                      [--gui] [--debug] [--info]

选项:
  --detector       检测后端 (默认: torchvision)
  --conf           检测置信度阈值 (默认: 0.5)
  --crf            x264 质量, 越小越好 (默认: 18)
  --preset         编码速度 preset (默认: slow)
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
- 严重遮挡（>60%）时会暂时退化到检测模式
- 飞机完全出画后需重新检测初始化
- CPU 版处理速度较慢（6K 视频约 2-4 fps）

## 技术栈

- **检测**：PyTorch Faster R-CNN (COCO 预训练)
- **追踪**：Sobel 梯度幅值 + NCC 模板匹配
- **I/O**：PyAV (FFmpeg)
- **GUI**：Tkinter

## 许可证

GNU General Public License v3
