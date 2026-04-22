from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
import datetime
from torchvision.ops import batched_nms
from tqdm import tqdm


# ====================== 优化：局部半透明掩码绘制函数 ======================
def draw_mask_on_roi(img, mask_roi, box, color=(0, 255, 0), alpha=0.5):
    """仅在边界框(ROI)区域内绘制掩码，速度提升百倍"""
    x1, y1, x2, y2 = map(int, box)

    # 提取原图 ROI
    roi = img[y1:y2, x1:x2]

    # 创建彩色掩码 ROI
    color_mask = np.zeros_like(roi, dtype=np.uint8)
    color_mask[mask_roi > 0] = color

    # 局部混合
    blended = cv2.addWeighted(color_mask, alpha, roi, 1.0, 0)

    # 仅更新掩码为前景的像素
    roi[mask_roi > 0] = blended[mask_roi > 0]
    img[y1:y2, x1:x2] = roi
    return img


# ====================== 核心：大图分块推理（优化版） ======================
def predict_large_image_optimized(
        model_path, image_path, output_dir,
        tile_size=640, overlap=0.2, conf_thres=0.25, iou_thres=0.45
):
    model = YOLO(model_path)
    print(f"✅ 模型加载成功: {model_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ 图像不存在: {image_path}")
    h, w = img.shape[:2]
    print(f"📐 原始图像尺寸: {w}×{h}")

    # 存储局部数据（防OOM策略）
    all_boxes = []
    all_scores = []
    all_classes = []
    all_cropped_masks = []  # 仅存储边界框内的二值掩码

    stride = int(tile_size * (1 - overlap))
    rows = (h - tile_size + stride) // stride + 1 if h >= tile_size else 1
    cols = (w - tile_size + stride) // stride + 1 if w >= tile_size else 1
    total_tiles = rows * cols

    print(f"🔄 开始分块处理: {rows}行 × {cols}列 = {total_tiles}个图块")

    # 1. 遍历切块推理
    with tqdm(total=total_tiles, desc="处理图块", unit="tile") as pbar:
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 直接切片，不需要手动 padding，YOLO 内部会处理 letterbox
                tile = img[y:y + tile_size, x:x + tile_size]
                actual_h, actual_w = tile.shape[:2]

                if tile.size == 0:
                    pbar.update(1)
                    continue

                # 单块推理
                results = model.predict(
                    source=tile, imgsz=tile_size, conf=conf_thres,
                    iou=iou_thres, task='segment', verbose=False
                )
                result = results[0]

                if result.masks is not None and len(result.masks) > 0:
                    # 提取 YOLO 内部张量
                    net_masks = result.masks.data.cpu().numpy()  # [N, H, W], float32
                    local_boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for i, mask_tensor in enumerate(net_masks):
                        # 将网络输出的特征图缩放回当前图块实际尺寸
                        mask_resized = cv2.resize(mask_tensor, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)

                        # 二值化（解决全黑 bug）
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)

                        # 获取当前目标的局部边界框，并限制在图块范围内
                        x1, y1, x2, y2 = map(int, local_boxes[i])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(actual_w, x2), min(actual_h, y2)

                        # 过滤掉异常的小框
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # 【核心防OOM】：仅裁剪并保存边界框内部的掩码
                        cropped_mask = mask_binary[y1:y2, x1:x2]

                        # 坐标转换为原图绝对坐标
                        global_box = [x + x1, y + y1, x + x2, y + y2]

                        all_cropped_masks.append(cropped_mask)
                        all_boxes.append(global_box)
                        all_scores.append(scores[i])
                        all_classes.append(classes[i])

                pbar.update(1)

    if not all_boxes:
        print("⚠️ 未检测到任何目标")
        return img, {}, np.zeros((h, w), dtype=np.uint8)

    # 2. 类别感知 NMS (Batched NMS)
    print("🔄 正在进行类别感知 NMS 去重...")
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    classes_tensor = torch.tensor(all_classes, dtype=torch.int64)  # batched_nms 要求 int64

    # batched_nms 确保不同类别的目标不会互相消除
    keep_indices = batched_nms(boxes_tensor, scores_tensor, classes_tensor, iou_thres).numpy()

    final_boxes = [all_boxes[i] for i in keep_indices]
    final_scores = [all_scores[i] for i in keep_indices]
    final_classes = [all_classes[i] for i in keep_indices]
    final_masks = [all_cropped_masks[i] for i in keep_indices]

    print(f"✅ NMS后保留 {len(keep_indices)}/{len(all_boxes)} 个目标")

    # 3. 统计类别与绘制
    print("🎨 正在极速绘制结果并生成全局掩码...")
    result_img = img.copy()
    global_combined_mask = np.zeros((h, w), dtype=np.uint8)
    class_counts = {}

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    for i in tqdm(range(len(keep_indices)), desc="渲染与合并", unit="obj"):
        box = final_boxes[i]
        cls_id = final_classes[i]
        conf = final_scores[i]
        cropped_mask = final_masks[i]
        x1, y1, x2, y2 = map(int, box)

        # 统计
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        # 极速绘制半透明掩码 (仅操作 ROI)
        color = colors[cls_id % len(colors)]
        result_img = draw_mask_on_roi(result_img, cropped_mask, box, color, alpha=0.45)

        # 还原全局掩码（拼图）
        global_combined_mask[y1:y2, x1:x2] = np.maximum(
            global_combined_mask[y1:y2, x1:x2], cropped_mask * 255
        )

        # 绘制文本与框 (可选：绘制边界框，如果不想要把 cv2.rectangle 注释掉)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"

        # 优化文本位置，防止出界
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        txt_y = y1 - 5 if y1 - 5 > th else y1 + th + 5
        cv2.putText(result_img, label, (x1, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 绘制统计面板
    y_offset = 30
    cv2.putText(result_img, f"Total: {len(keep_indices)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(result_img, f"Total: {len(keep_indices)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    y_offset += 35
    for cls_name, count in class_counts.items():
        cv2.putText(result_img, f"{cls_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(result_img, f"{cls_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)
        y_offset += 30

    print(f"✅ 处理完成 | 类别统计: {class_counts}")
    return result_img, class_counts, global_combined_mask


# ====================== 主函数 ======================
def main():
    # ---------- 配置区 ----------
    MODEL_PATH = r"G:\YOLO\ultralytics-8.3.163\runs\segment\train17\weights\best.pt"
    IMAGE_PATH = r"C:\Users\10761\Desktop\毕业设计\3.png"
    BASE_OUTPUT_DIR = "runs/segment"
    TILE_SIZE = 640
    OVERLAP = 0.2
    # ----------------------------

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"large_image_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        result_img, stats, mask = predict_large_image_optimized(
            model_path=MODEL_PATH,
            image_path=IMAGE_PATH,
            output_dir=output_dir,
            tile_size=TILE_SIZE,
            overlap=OVERLAP
        )

        base_name = os.path.basename(IMAGE_PATH)
        name_without_ext = os.path.splitext(base_name)[0]

        cv2.imwrite(os.path.join(output_dir, f"result_{base_name}"), result_img)
        cv2.imwrite(os.path.join(output_dir, f"mask_{base_name}"), mask)  # mask 已经是 0和255

        stats_path = os.path.join(output_dir, f"stats_{name_without_ext}.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=== 检测结果 ===\n")
            for cls_name, count in stats.items():
                f.write(f"{cls_name}: {count}\n")
            f.write(f"总计: {sum(stats.values())}\n")

        print(f"💾 所有结果已保存至: {output_dir}")

        # 缩放显示 (避免原图太大屏幕放不下)
        show_img = result_img.copy()
        h, w = show_img.shape[:2]
        if max(h, w) > 1080:
            scale = 1080 / max(h, w)
            show_img = cv2.resize(show_img, (int(w * scale), int(h * scale)))

        cv2.imshow("Optimized Segmentation", show_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()