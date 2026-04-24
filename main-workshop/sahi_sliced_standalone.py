import cv2
import numpy as np
import torch
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def run(model_path, image_path, tile_size=640, overlap_ratio=0.25):
    # 1. 调整参数：温和的 NMS 避免误杀紧密种子
    CONF_THRES = 0.20  # 略微降低置信度，把粘连的边缘种子找出来
    MATCH_THRES = 0.45  # 提高去重阈值(原为0.4)。意味着只有两个框重合度超过65%才会被当成同一个删掉

    # 2. 引入形态过滤参数（根据你的图片分辨率微调）
    MIN_AREA = 50  # 过滤掉太小的幽灵噪点 (面积小于50像素的不要)
    MAX_AREA = 3800  # 过滤掉异常大的背景误判 (面积大于3000像素的不要)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=CONF_THRES,
        device=device
    )

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=MATCH_THRES  # 使用调高后的阈值
    )

    img = cv2.imread(image_path)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    class_counts = {}

    valid_count = 0  # 记录过滤后的真实种子数量

    for pred in result.object_prediction_list:
        # 提取类别
        cls_id = int(pred.category.id)
        cls_name = pred.category.name
        color = colors[cls_id % len(colors)]

        # --- 核心新增：基于掩码的过滤逻辑 ---
        if pred.mask is not None:
            bool_mask = np.array(pred.mask.bool_mask, dtype=bool)

            # 计算这个掩码的真实像素面积
            mask_area = np.count_nonzero(bool_mask)

            # 【拦截器】：如果面积太小（噪点）或太大（假阳性幽灵），直接跳过不画！
            if mask_area < MIN_AREA or mask_area > MAX_AREA:
                continue

            # 计算质心
            M = cv2.moments(bool_mask.astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue  # 没有质心的直接丢弃

            # 渲染被保留下来的有效种子
            img[bool_mask] = (img[bool_mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        else:
            # 如果模型吐出了结果但没有Mask（通常是假阳性），直接跳过
            continue

        valid_count += 1
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        # 在质心绘制数字编号
        cv2.putText(img, str(valid_count), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, str(valid_count), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 绘制统计信息框
    overlay = img.copy()
    box_h = 70 + len(class_counts) * 35
    cv2.rectangle(overlay, (10, 10), (320, box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    cv2.putText(img, f"TOTAL: {valid_count}", (25, 50), 2, 0.9, (255, 255, 255), 2)
    for idx, (name, count) in enumerate(class_counts.items()):
        cv2.putText(img, f"{name}: {count}", (30, 90 + idx * 35), 2, 0.7, (0, 255, 0), 2)

    return img, valid_count, class_counts


if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = r"G:\YOLO\ultralytics-8.3.163\runs\segment\train17\weights\best.pt"
    IMAGE_PATH = r"C:\Users\10761\Desktop\毕业设计\3.png"
    OUTPUT_DIR = "runs/yolo11_sahi_output"

    print("开始使用 SAHI 进行切片推理...")
    # 可调节 overlap_ratio (推荐 0.2 到 0.25)，以确保小目标能完整落入切片中
    result_img, total, class_counts = run(MODEL_PATH, IMAGE_PATH, tile_size=640, overlap_ratio=0.2)

    print(f"\n检测完成！共检测到 {total} 粒种子。")
    print(f"分类统计: {class_counts}")

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "final_sahi_res.jpg")
    cv2.imwrite(save_path, result_img)
    print(f"结果图片已保存至: {save_path}")

    # 缩放窗口显示结果（防止原图过大超出屏幕）
    h_s, w_s = result_img.shape[:2]
    # 固定高度为1000像素，按比例缩放宽度
    scale = 1000 / h_s
    cv2.imshow("SAHI Segmentation Result", cv2.resize(result_img, (int(w_s * scale), 1000)))

    # 按任意键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()