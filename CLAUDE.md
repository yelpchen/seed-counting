# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A YOLOv11-based seed counting system. It uses instance segmentation to detect and count seeds in images via a sliced inference approach. The project has two entry points: a PyQt6 GUI and a CLI script.

## Running the Application

```bash
# GUI (primary entry point)
python pyqt/main.py

# CLI inference
python main-workshop/mymask.py
```

## Key Dependencies

Install manually (no top-level requirements.txt):
```bash
pip install ultralytics torch torchvision opencv-python PyQt6 numpy pillow
```

The `main-workshop/` directory contains a vendored copy of the Ultralytics library — do not confuse it with the installed package.

## Architecture

### Inference pipeline (`main-workshop/mymask.py`)

`run(model_path, image_path)` is the core function:
1. Divides the input image into 640×640 tiles with 100px overlap (`_manual_sliced_predict`)
2. Runs YOLOv11-seg on each tile, collecting boxes + masks
3. Applies `torchvision.ops.batched_nms` across all tiles to remove duplicates
4. Renders semi-transparent colored masks and numbered labels at mask centroids
5. Returns `(annotated_bgr_ndarray, total_count)`

### GUI (`pyqt/main.py`)

`SeedCountingUI` (QMainWindow) with:
- Left sidebar: model selector (scans `pyqt/models/*.pt`), run button, image tools
- Right panel: `QGraphicsView` with scroll/zoom/rotate support
- `InferenceThread(QThread)` runs `mymask.run()` off the main thread; emits `finished(annotated, count)` or `error(str)`

The GUI's `InferenceThread` adds `main-workshop/` to `sys.path` at runtime to import `mymask`.

### Model storage

- `pyqt/models/` — models used by the GUI (scanned at startup)
- `main-workshop/models/` — models used by CLI scripts (e.g., `seed3.pt`)

### Training

`main-workshop/mytrain.py` trains from `yolo11n-seg.pt` using a dataset config `myseed.yaml` (not committed). Outputs go to `runs/segment/`.

## File Layout

```
pyqt/main.py          # GUI entry point
main-workshop/
  mymask.py           # core inference engine (import this for new scripts)
  mycount.py          # alternative counter with transparent text
  mytrain.py          # training script
  models/             # CLI model weights
  datasets/           # sample data
  ultralytics/        # vendored Ultralytics source
```