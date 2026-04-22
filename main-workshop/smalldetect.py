from ultralytics import YOLO
import cv2
import numpy as np
import os


def draw_transparent_text(img, text, position, font, font_scale, color, thickness, alpha=0.5):
    """
    在图像上绘制半透明文本
    :param img: 原始图像（BGR格式，会被修改）
    :param text: 文本内容
    :param position: 文本左下角坐标 (x, y)
    :param font: 字体类型（如 cv2.FONT_HERSHEY_SIMPLEX）
    :param font_scale: 字体大小
    :param color: 文本颜色（BGR）
    :param thickness: 线条粗细
    :param alpha: 文本透明度（0完全透明，1完全不透明）
    """
    # 获取文本尺寸
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # 文本区域边界框（增加一些内边距）
    x, y = position
    rect_x1 = x
    rect_y1 = y - text_h - baseline  # 文字区域左上角
    rect_x2 = x + text_w
    rect_y2 = y + baseline

    # 确保坐标在图像范围内
    rect_x1 = max(0, rect_x1)
    rect_y1 = max(0, rect_y1)
    rect_x2 = min(img.shape[1], rect_x2)
    rect_y2 = min(img.shape[0], rect_y2)

    # 提取原图对应区域
    roi = img[rect_y1:rect_y2, rect_x1:rect_x2].copy()

    # 创建临时图像（黑色背景），在该区域绘制文本
    temp = np.zeros_like(roi)
    cv2.putText(temp, text, (x - rect_x1, y - rect_y1), font, font_scale, color, thickness)

    # 混合：临时图像（文本）与原图ROI
    blended = cv2.addWeighted(temp, alpha, roi, 1 - alpha, 0)

    # 将混合后的区域放回原图
    img[rect_y1:rect_y2, rect_x1:rect_x2] = blended


def main():
    # 加载模型
    model = YOLO(r"G:\YOLO\ultralytics-8.3.163\runs\segment\train17\weights\best.pt")

    # 预测，不自动保存，使用流式处理以便逐张图像操作
    results = model.predict(
        source=r"G:\YOLO\ultralytics-8.3.163\image\3\images",
        save=False,
        stream=True,  # 流式返回，便于逐张处理
        show=False,
    )

    # 设置输出目录（YOLO默认风格）
    output_dir = model.predictor.save_dir if hasattr(model.predictor, 'save_dir') else "runs/detect/predict"
    os.makedirs(output_dir, exist_ok=True)

    # 字体参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = (255, 255, 255)  # 白色
    box_color = (0, 255, 0)  # 绿色边框
    alpha = 0.5  # 透明度50%

    for i, result in enumerate(results):
        # 原始图像（BGR）
        img = result.orig_img.copy()
        # 检测框信息
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # 统计各类别数量
            class_counts = {}
            for cls in boxes.cls:
                class_name = result.names[int(cls)]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # 绘制检测框和框内编号
            for idx, box in enumerate(boxes, start=1):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]

                # 绘制边框
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)

                # 在框内显示目标编号
                text = str(idx)
                text_size = cv2.getTextSize(text, font, 0.8, 2)[0]

                # 计算文本位置（框的中心）
                text_x = xyxy[0] + (xyxy[2] - xyxy[0] - text_size[0]) // 2
                text_y = xyxy[1] + (xyxy[3] - xyxy[1] + text_size[1]) // 2

                # 绘制黑色背景
                padding = 5
                cv2.rectangle(
                    img,
                    (text_x - padding, text_y - text_size[1] - padding),
                    (text_x + text_size[0] + padding, text_y + padding),
                    (0, 0, 0),
                    -1
                )

                # 绘制白色数字
                cv2.putText(img, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)

            # 绘制计数文本（左上角）
            if class_counts:
                y_offset = 30  # 起始Y坐标
                for class_name, count in class_counts.items():
                    count_text = f"{class_name}: {count}"
                    draw_transparent_text(img, count_text, (10, y_offset), font, font_scale, text_color, thickness,
                                          alpha)
                    y_offset += 30  # 行间距

        # 保存图像
        save_path = os.path.join(output_dir, f"result_{i:06d}.jpg")
        cv2.imwrite(save_path, img)

    print(f"所有结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()