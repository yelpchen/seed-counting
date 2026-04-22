from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
from torchvision.ops import nms


# ====================== 半透明掩码绘制函数 ======================
def draw_transparent_mask(img, mask, color=(0, 255, 0), alpha=0.5):
    """在图像上绘制半透明掩码"""
    # 创建彩色掩码
    color_mask = np.zeros_like(img, dtype=np.uint8)
    color_mask[mask > 0] = color

    # 混合原图和彩色掩码
    blended = cv2.addWeighted(color_mask, alpha, img, 1 - alpha, 0)
    img[mask > 0] = blended[mask > 0]
    return img


# ====================== 核心：大图分块推理（带掩码） ======================
def predict_large_image_with_masks(
        model_path,
        image_path,
        tile_size=640,
        overlap=0.2,
        conf_thres=0.25,
        iou_thres=0.45
):
    # 1. 加载模型
    model = YOLO(model_path)
    print(f"✅ 模型加载成功: {model_path}")

    # 2. 读取大图
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ 图像不存在: {image_path}")
    h, w = img.shape[:2]
    print(f"📐 原始图像尺寸: {w}×{h}")

    # 3. 准备存储所有掩码
    all_masks = []
    all_scores = []
    all_classes = []
    all_boxes = []

    # 4. 滑动窗口分块推理
    stride = int(tile_size * (1 - overlap))

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 提取图块
            tile = img[y:y + tile_size, x:x + tile_size]
            if tile.size == 0:
                continue

            # 填充边缘块
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                pad_tile = np.full((tile_size, tile_size, 3), 114, dtype=np.uint8)
                pad_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad_tile

            # 单块推理（启用分割）
            results = model.predict(
                source=tile,
                imgsz=tile_size,
                conf=conf_thres,
                iou=iou_thres,
                task='segment',  # 明确指定为分割任务
                verbose=False
            )
            result = results[0]

            # 收集检测结果
            if result.masks is not None and len(result.masks) > 0:
                # 获取掩码（已调整为图块大小）
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                # 坐标映射回原图
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y

                # 存储掩码信息
                for i, mask in enumerate(masks):
                    # 调整掩码到原图位置
                    full_mask = np.zeros((h, w), dtype=np.uint8)

                    # 计算实际图块区域（考虑填充）
                    actual_h = min(tile_size, h - y)
                    actual_w = min(tile_size, w - x)

                    # 将掩码放置到正确位置
                    mask_resized = cv2.resize(mask.astype(np.uint8), (actual_w, actual_h))
                    full_mask[y:y + actual_h, x:x + actual_w] = mask_resized

                    all_masks.append(full_mask)
                    all_boxes.append(boxes[i])
                    all_scores.append(scores[i])
                    all_classes.append(classes[i])

    # 5. 如果没有检测到任何目标
    if not all_masks:
        print("⚠️ 未检测到任何目标")
        return img, {}

    # 6. 合并重叠掩码（简化版：直接叠加）
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in all_masks:
        combined_mask = np.maximum(combined_mask, mask)

    # 7. 统计类别数量
    class_counts = {}
    for cls_id in all_classes:
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    # 8. 绘制掩码到原图（不绘制边框）
    result_img = img.copy()

    # 为每个类别分配不同颜色
    colors = [
        (0, 255, 0),  # 绿色
        (255, 0, 0),  # 蓝色
        (0, 0, 255),  # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 黄色
    ]

    # 绘制每个掩码
    for i, (mask, cls_id) in enumerate(zip(all_masks, all_classes)):
        color = colors[cls_id % len(colors)]
        result_img = draw_transparent_mask(result_img, mask, color, alpha=0.4)

        # 在掩码中心显示类别和置信度
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            cls_name = model.names[cls_id]
            conf = all_scores[i]
            label = f"{cls_name}: {conf:.2f}"

            cv2.putText(result_img, label, (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 9. 绘制类别统计
    y_offset = 30
    for cls_name, count in class_counts.items():
        cv2.putText(result_img, f"{cls_name}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    print(f"✅ 分割完成 | 总目标数: {len(all_masks)} | 类别统计: {class_counts}")
    return result_img, class_counts, combined_mask


# ====================== 主函数 ======================
def main():
    # ---------- 配置区 ----------
    MODEL_PATH = r"G:\YOLO\ultralytics-8.3.163\runs\segment\train16\weights\best.pt"
    IMAGE_PATH = r"C:\Users\10761\Desktop\毕业设计\3.png"
    OUTPUT_DIR = "runs/segment/large_image"
    TILE_SIZE = 640
    OVERLAP = 0.2
    # ----------------------------

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 执行大图分割推理
        result_img, stats, mask = predict_large_image_with_masks(
            model_path=MODEL_PATH,
            image_path=IMAGE_PATH,
            tile_size=TILE_SIZE,
            overlap=OVERLAP
        )

        # 保存结果
        base_name = os.path.basename(IMAGE_PATH)
        save_path = os.path.join(OUTPUT_DIR, base_name)
        cv2.imwrite(save_path, result_img)

        # 保存掩码
        mask_path = os.path.join(OUTPUT_DIR, f"mask_{base_name}")
        cv2.imwrite(mask_path, mask * 255)  # 转换为0-255范围

        print(f"💾 结果已保存: {save_path}")
        print(f"💾 掩码已保存: {mask_path}")

        # 显示结果
        cv2.imshow("Segmentation Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()