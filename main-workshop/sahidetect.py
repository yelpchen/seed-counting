import os
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import torch
from tqdm import tqdm
import time

# =========================================================
# 1. 基础配置
# =========================================================
IMAGE_PATH = r"C:\Users\10761\Desktop\part-00660-2127.jpg"  # 替换为你需要检测的大图路径
MODEL_PATH = r"G:\YOLO\seed-counting\main-workshop\yolo11n-seg.pt"  # 替换为你的 YOLOv11Seg 权重
OUTPUT_DIR = r"runs/detect/sahi_result"
CONFIDENCE_THRESHOLD = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. 加载模型
# =========================================================
print("🚀 正在加载模型...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",  # 使用 yolov8 适配器
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=device
    )
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

# =========================================================
# 3. 切片推理
# =========================================================
print("🔪 开始切片推理...")

image = read_image(IMAGE_PATH)
height, width = image.shape[:2]

# 切片参数
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_H_RATIO = 0.2
OVERLAP_W_RATIO = 0.2

# 计算切片数量
step_h = int(SLICE_HEIGHT * (1 - OVERLAP_H_RATIO))
step_w = int(SLICE_WIDTH * (1 - OVERLAP_W_RATIO))
num_slices_h = max(1, (height - SLICE_HEIGHT + step_h - 1) // step_h + 1)
num_slices_w = max(1, (width - SLICE_WIDTH + step_w - 1) // step_w + 1)
total_slices = num_slices_h * num_slices_w

print(f"📐 图像尺寸: {width}x{height}")
print(f"📦 预计切片数量: {total_slices}")

start_time = time.time()

result = get_sliced_prediction(
    image=image,
    detection_model=detection_model,
    slice_height=SLICE_HEIGHT,
    slice_width=SLICE_WIDTH,
    overlap_height_ratio=OVERLAP_H_RATIO,
    overlap_width_ratio=OVERLAP_W_RATIO,
    verbose=0
)

end_time = time.time()
print(f"⏱️ 推理耗时: {end_time - start_time:.2f} 秒")

# =========================================================
# 4. 计数统计
# =========================================================
print("\n📊 统计结果...")

object_counts = {}
prediction_list = result.object_prediction_list

if len(prediction_list) == 0:
    print("⚠️ 警告：未检测到任何目标！")
else:
    for pred in tqdm(prediction_list, desc="🔢 计数中", unit="obj"):
        name = pred.category.name or f"Class_{pred.category.id}"
        object_counts[name] = object_counts.get(name, 0) + 1

    print("-" * 40)
    for cls_name, cnt in sorted(object_counts.items()):
        print(f"  📌 {cls_name}: {cnt}")
    print(f"🔥 总计: {len(prediction_list)}")
    print("-" * 40)

# =========================================================
# 5. 可视化并保存结果图片（修复所有错误）
# =========================================================
print("🖼️ 正在生成可视化结果...")

# 方法1：尝试导出可视化
output_image = result.export_visuals(
    export_dir=OUTPUT_DIR,
    text_size=0.5,
    rect_th=2,
    hide_labels=False,
    hide_conf=False
)

# 方法2：如果返回 None，手动绘制
if output_image is None:
    print("⚠️ export_visuals 返回空，手动绘制检测框...")
    output_image = image.copy()

    # 手动绘制每个检测框 - 修复 BoundingBox API
    for pred in prediction_list:
        try:
            # 新版 SAHI 使用直接属性访问
            bbox = pred.bbox
            x1 = int(bbox.minx)
            y1 = int(bbox.miny)
            x2 = int(bbox.maxx)
            y2 = int(bbox.maxy)

            # 绘制矩形框
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 如果有类别名称，写上标签
            label = pred.category.name or f"Class_{pred.category.id}"
            conf = pred.score.value if hasattr(pred.score, 'value') else pred.score
            label_text = f"{label}: {conf:.2f}"

            # 确保标签不会超出图像边界
            text_y = max(y1 - 10, 20)
            cv2.putText(output_image, label_text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"⚠️ 绘制单个检测框时出错: {e}")
            continue

# 保存图片（注意：read_image 返回 RGB，cv2 需要 BGR）
output_path = os.path.join(OUTPUT_DIR, "result.jpg")
try:
    if output_image is not None:
        # 转换 RGB -> BGR
        if len(output_image.shape) == 3 and output_image.shape[2] == 3:
            output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        else:
            output_image_bgr = output_image
        cv2.imwrite(output_path, output_image_bgr)
        print(f"✅ 检测结果已保存到: {output_path}")
    else:
        print("❌ 无法生成可视化图像")
except Exception as e:
    print(f"❌ 保存图像失败: {e}")

print("\n🎉 处理完成！")