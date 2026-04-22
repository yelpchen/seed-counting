from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch
from torchvision.ops import nms  # 改用 torchvision 的 NMS


# ====================== 半透明文本绘制函数 ======================
def draw_transparent_text(img, text, position, font, font_scale, color, thickness, alpha=0.5):
    """在图像上绘制半透明文本"""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    rect_x1 = max(0, x)
    rect_y1 = max(0, y - text_h - baseline)
    rect_x2 = min(img.shape[1], x + text_w)
    rect_y2 = min(img.shape[0], y + baseline)

    if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
        return

    roi = img[rect_y1:rect_y2, rect_x1:rect_x2].copy()
    temp = np.zeros_like(roi)
    cv2.putText(temp, text, (x - rect_x1, y - rect_y1), font, font_scale, color, thickness)
    blended = cv2.addWeighted(temp, alpha, roi, 1 - alpha, 0)
    img[rect_y1:rect_y2, rect_x1:rect_x2] = blended


# ====================== 核心：大图分块推理 ======================
def predict_large_image(
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

    # 3. 分块参数
    stride = int(tile_size * (1 - overlap))
    all_boxes, all_scores, all_classes = [], [], []

    # 4. 滑动窗口分块推理
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

            # 单块推理
            results = model.predict(
                source=tile,
                imgsz=tile_size,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )
            result = results[0]

            # 收集检测结果
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                # 坐标映射回原图
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)

    # 5. 合并所有块的结果并NMS去重
    if not all_boxes:
        print("⚠️ 未检测到任何目标")
        return img, {}

    # 转换为张量
    all_boxes_tensor = torch.from_numpy(np.concatenate(all_boxes))
    all_scores_tensor = torch.from_numpy(np.concatenate(all_scores))
    all_classes_np = np.concatenate(all_classes)

    # 使用 torchvision 的 NMS（更稳定）
    keep_indices = nms(all_boxes_tensor, all_scores_tensor, iou_thres)

    final_boxes = all_boxes_tensor[keep_indices].numpy()
    final_scores = all_scores_tensor[keep_indices].numpy()
    final_classes = all_classes_np[keep_indices.numpy()]

    # 6. 统计类别数量
    class_counts = {}
    for cls_id in final_classes:
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    # 7. 绘制结果到原图
    for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
        x1, y1, x2, y2 = box.astype(int)
        cls_name = model.names[cls_id]

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制半透明标签
        label = f"{cls_name} {score:.2f}"
        draw_transparent_text(img, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (255, 255, 255), 2, 0.5)

    # 绘制类别统计
    y_offset = 30
    for cls_name, count in class_counts.items():
        draw_transparent_text(img, f"{cls_name}: {count}", (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (255, 255, 255), 2, 0.5)
        y_offset += 30

    print(f"✅ 检测完成 | 总目标数: {len(final_boxes)} | 类别统计: {class_counts}")
    return img, class_counts


# ====================== 主函数 ======================
def main():
    # ---------- 配置区 ----------
    MODEL_PATH = r"G:\YOLO\ultralytics-8.3.163\runs\segment\train17\weights\best.pt"
    IMAGE_PATH = r"C:\Users\10761\Desktop\毕业设计\3.png"
    OUTPUT_DIR = "runs/detect/large_image"
    TILE_SIZE = 640
    OVERLAP = 0.2
    # ----------------------------

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 执行大图推理
        result_img, stats = predict_large_image(
            model_path=MODEL_PATH,
            image_path=IMAGE_PATH,
            tile_size=TILE_SIZE,
            overlap=OVERLAP
        )

        # 保存结果
        save_path = os.path.join(OUTPUT_DIR, os.path.basename(IMAGE_PATH))
        cv2.imwrite(save_path, result_img)
        print(f"💾 结果已保存: {save_path}")

        # 显示结果（可选）
        cv2.imshow("Detection Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Windows多进程保护
    torch.multiprocessing.freeze_support()
    main()