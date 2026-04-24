# Seed Counting

基于 YOLOv11 实例分割的种子计数系统，支持切片推理与直接推理两种模式，提供 PyQt6 图形界面，可用于种子图像的自动检测、分类统计与可视化标注。

## 项目内容

- 使用 YOLOv11-seg 对种子进行实例分割
- 支持两种推理模式：
  - **切片推理（sliced）**：适合小目标密集场景
  - **直接推理（direct）**：适合目标较大场景
- 输出内容：
  - 叠加掩码与编号后的结果图
  - 总数统计
  - 分类统计表
- 提供桌面 GUI（PyQt6）进行模型选择、图像加载、缩放/旋转与一键检测

## 项目构成

```text
seed-counting/
├─ main-workshop/
│  ├─ main.py                # GUI 主入口（当前打包入口）
│  ├─ mymask.py              # 核心推理逻辑（切片/直接推理）
│  ├─ models/                # 模型权重目录（.pt）
│  ├─ icon/                  # 图标资源（favicon.ico / seed.png）
│  └─ ultralytics/           # 项目内置 ultralytics 源码
├─ pyqt/
│  ├─ main.py                # 另一版 GUI 入口
│  └─ models/                # GUI 使用的模型目录
├─ seed-counting.spec        # PyInstaller 打包配置
└─ build.bat                 # 简单打包脚本
```

## 运行方式

### 1) 图形界面（main-workshop）

```bash
python main-workshop/main.py
```

### 2) 图形界面（pyqt）

```bash
python pyqt/main.py
```

## 环境依赖

建议 Python 3.10。

```bash
pip install ultralytics torch torchvision opencv-python PyQt6 numpy pillow
```

## 在 Anaconda 环境下使用 PyInstaller 打包

### 1) 创建并激活环境

```bash
conda create -n seed-pack python=3.10 -y
conda activate seed-pack
```

### 2) 安装依赖与打包工具

```bash
pip install -U pip
pip install pyinstaller ultralytics torch torchvision opencv-python PyQt6 numpy pillow
```

### 3) 进入项目根目录并打包

```bash
cd /g/YOLO/seed-counting
pyinstaller seed-counting.spec --clean
```

### 4) 打包结果

可执行文件输出目录：

```text
dist/种子计数系统/种子计数系统.exe
```

## 打包说明（当前 spec 已处理）

- 打包入口：`main-workshop/main.py`
- 自动携带资源：
  - `main-workshop/models`
  - `main-workshop/icon`
  - `main-workshop/ultralytics`
- EXE 图标：`main-workshop/icon/favicon.ico`
- 运行时兼容源码/打包环境图标路径（包含 `_MEIPASS` 场景）
