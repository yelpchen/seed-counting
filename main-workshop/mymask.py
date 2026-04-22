import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO


def _manual_sliced_predict(img, model, tile_size, overlap, conf_thres, iou_thres):
    h, w = img.shape[:2]
    all_boxes, all_scores, all_cls, all_masks = [], [], [], []
    stride = tile_size - overlap

    for y1 in range(0, h, stride):
        for x1 in range(0, w, stride):
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)
            tile = img[y1:y2, x1:x2]
            res = model.predict(tile, conf=conf_thres, verbose=False, task='segment')[0]

            if res.masks is not None:
                boxes = res.boxes.xyxy.cpu().numpy().copy()
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                all_boxes.append(boxes)
                all_scores.append(res.boxes.conf.cpu().numpy())
                all_cls.append(res.boxes.cls.cpu().numpy())
                for i in range(len(res.masks.data)):
                    m = res.masks.data[i].cpu().numpy()
                    m_resized = cv2.resize(m, (x2 - x1, y2 - y1))
                    all_masks.append((m_resized > 0.5, (y1, y2, x1, x2)))

    if not all_boxes:
        return np.array([]), {}, [], [], []

    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_cls = np.concatenate(all_cls)

    from torchvision.ops import batched_nms
    keep = batched_nms(
        torch.tensor(all_boxes), torch.tensor(all_scores),
        torch.tensor(all_cls, dtype=torch.int64), iou_thres
    ).numpy()

    return keep, all_boxes, all_scores, all_cls, all_masks


def run(model_path, image_path, tile_size=640, overlap=100, conf_thres=0.25, iou_thres=0.45):
    """切片推理 + 掩码渲染，返回 (annotated_bgr, total_count)"""
    model = YOLO(model_path)
    img = cv2.imread(image_path)

    keep_idx, all_boxes, all_scores, all_cls, all_masks = _manual_sliced_predict(
        img, model, tile_size, overlap, conf_thres, iou_thres
    )

    if len(keep_idx) == 0:
        return img, 0

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    class_counts = {}

    for count_id, i in enumerate(keep_idx, start=1):
        box = all_boxes[i]
        cls_id = int(all_cls[i])
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        color = colors[cls_id % len(colors)]

        m_data, (y1, y2, x1, x2) = all_masks[i]
        roi = img[y1:y2, x1:x2]
        roi[m_data] = (roi[m_data] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        img[y1:y2, x1:x2] = roi

        M = cv2.moments(m_data.astype(np.uint8))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
        else:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

        cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 左上角看板
    overlay = img.copy()
    box_h = 70 + len(class_counts) * 35
    cv2.rectangle(overlay, (10, 10), (320, box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, f"TOTAL: {len(keep_idx)}", (25, 50), 2, 0.9, (255, 255, 255), 2)
    for idx, (name, count) in enumerate(class_counts.items()):
        cv2.putText(img, f"{name}: {count}", (30, 90 + idx * 35), 2, 0.7, (0, 255, 0), 2)

    return img, len(keep_idx), class_counts


def run_direct(model_path, image_path, conf_thres=0.25, iou_thres=0.45):
    """直接推理（不切片），适合大目标，返回 (annotated_bgr, total_count)"""
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    res = model.predict(img, conf=conf_thres, iou=iou_thres, verbose=False, task='segment')[0]

    if res.masks is None:
        return img, 0, {}

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    h, w = img.shape[:2]
    class_counts = {}

    for count_id, i in enumerate(range(len(res.masks.data)), start=1):
        cls_id = int(res.boxes.cls[i].item())
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        color = colors[cls_id % len(colors)]

        m = res.masks.data[i].cpu().numpy()
        m = cv2.resize(m, (w, h)) > 0.5
        img[m] = (img[m] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        M = cv2.moments(m.astype(np.uint8))
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            box = res.boxes.xyxy[i].cpu().numpy()
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

        cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    total = len(res.masks.data)
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 70), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, f"TOTAL: {total}", (25, 50), 2, 0.9, (255, 255, 255), 2)
    return img, total, class_counts


if __name__ == "__main__":
    MODEL_PATH = r"G:\YOLO\ultralytics-8.3.163\runs\segment\train17\weights\best.pt"
    IMAGE_PATH = r"C:\Users\10761\Desktop\毕业设计\3.png"
    OUTPUT_DIR = "runs/yolo11_manual_output"

    result_img, total = run(MODEL_PATH, IMAGE_PATH)
    print(f"检测到 {total} 粒种子")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "final_manual_res.jpg")
    cv2.imwrite(save_path, result_img)
    print(f"保存完成: {save_path}")

    h_s, w_s = result_img.shape[:2]
    scale = 1000 / h_s
    cv2.imshow("Result", cv2.resize(result_img, (int(w_s * scale), 1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
