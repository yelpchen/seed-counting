import cv2
import numpy as np
import torch
import os
import datetime
from ultralytics import YOLO
from tqdm import tqdm

# ====================== 1. 配置区 ======================
MODEL_PATH = r"G:\YOLO\ultralytics-8.3.163\runs\segment\train17\weights\best.pt"
IMAGE_PATH = r"C:\Users\10761\Desktop\毕业设计\3.png"
OUTPUT_DIR = "runs/yolo11_manual_output"

TILE_SIZE = 640  # 每个切块的大小
OVERLAP = 100  # 切块重叠像素
CONF_THRES = 0.25
IOU_THRES = 0.45  # NMS 阈值

# ====================== 2. 加载模型 ======================
print("⏳ 正在加载 YOLOv11 原生模型...")
model = YOLO(MODEL_PATH)


# ====================== 3. 手动切片预测逻辑 ======================
def manual_sliced_predict(img_path, model, tile_size=640, overlap=100):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    all_boxes = []
    all_scores = []
    all_cls = []
    all_masks = []

    # 计算步长
    stride = tile_size - overlap

    # 生成切块坐标
    for y1 in range(0, h, stride):
        for x1 in range(0, w, stride):
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            # 切片
            tile = img[y1:y2, x1:x2]

            # 推理
            results = model.predict(tile, conf=CONF_THRES, verbose=False, task='segment')
            res = results[0]

            if res.masks is not None:
                # 转换坐标到原图
                boxes = res.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] += x1  # 补偿 X 偏移
                boxes[:, [1, 3]] += y1  # 补偿 Y 偏移

                all_boxes.append(boxes)
                all_scores.append(res.boxes.conf.cpu().numpy())
                all_cls.append(res.boxes.cls.cpu().numpy())

                # 处理掩码：缩放到原图相应位置
                # 注意：这里为了防OOM，我们暂存局部掩码和位置
                for i in range(len(res.masks.data)):
                    m = res.masks.data[i].cpu().numpy()
                    m_resized = cv2.resize(m, (x2 - x1, y2 - y1))
                    # 存储为 (二值掩码, (y1, y2, x1, x2))
                    all_masks.append((m_resized > 0.5, (y1, y2, x1, x2)))

    if not all_boxes:
        return img, 0, {}

    # 合并结果
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_cls = np.concatenate(all_cls)

    # 4. 执行 NMS (使用 torchvision 提高速度)
    from torchvision.ops import batched_nms
    keep = batched_nms(torch.tensor(all_boxes), torch.tensor(all_scores), torch.tensor(all_cls), IOU_THRES)

    return img, keep.numpy(), (all_boxes, all_scores, all_cls, all_masks)


# ====================== 4. 运行预测与渲染 ======================
print(f"🚀 正在处理大图: {IMAGE_PATH}")
img, keep_idx, data = manual_sliced_predict(IMAGE_PATH, model, TILE_SIZE, OVERLAP)

if len(keep_idx) == 0:
    print("❌ 未检测到目标")
    exit()

all_boxes, all_scores, all_cls, all_masks = data
class_counts = {}
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]

print(f"🎨 正在渲染结果，目标总数: {len(keep_idx)}")

for count_id, i in enumerate(tqdm(keep_idx, desc="渲染中"), start=1):
    box = all_boxes[i]
    cls_id = int(all_cls[i])
    cls_name = model.names[cls_id]
    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    color = colors[cls_id % len(colors)]

    # 提取并绘制掩码
    m_data, (y1, y2, x1, x2) = all_masks[i]

    # 局部染色
    roi = img[y1:y2, x1:x2]
    roi[m_data] = (roi[m_data] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    img[y1:y2, x1:x2] = roi

    # 计算中心点写编号
    M = cv2.moments(m_data.astype(np.uint8))
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
    else:
        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

    cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# ====================== 5. 绘制看板 ======================
overlay = img.copy()
box_h = 70 + len(class_counts) * 35
cv2.rectangle(overlay, (10, 10), (320, box_h), (0, 0, 0), -1)
img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

cv2.putText(img, f"TOTAL: {len(keep_idx)}", (25, 50), 2, 0.9, (255, 255, 255), 2)
for i, (name, count) in enumerate(class_counts.items()):
    cv2.putText(img, f"{name}: {count}", (30, 90 + i * 35), 2, 0.7, (0, 255, 0), 2)

# ====================== 6. 保存与显示 ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "final_manual_res.jpg")
cv2.imwrite(save_path, img)
print(f"✅ 保存完成: {save_path}")

# 缩放预览
h_s, w_s = img.shape[:2]
scale = 1000 / h_s
cv2.imshow("YOLOv11 Manual Sliced Result", cv2.resize(img, (int(w_s * scale), 1000)))
cv2.waitKey(0)
cv2.destroyAllWindows()