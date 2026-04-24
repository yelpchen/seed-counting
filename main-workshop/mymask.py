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


def _seam_positions(length, stride):
    if stride <= 0:
        return []
    return [p for p in range(stride, length, stride)]


def _mask_centroid(mask, x1, y1, box):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] != 0:
        return int(M["m10"] / M["m00"]) + x1, int(M["m01"] / M["m00"]) + y1
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)


def _touching_seams(mask, roi, seam_xs, seam_ys, seam_band):
    y1, y2, x1, x2 = roi
    seams = set()

    for sx in seam_xs:
        if sx < x1 - seam_band or sx > x2 + seam_band:
            continue
        lx0 = max(0, sx - seam_band - x1)
        lx1 = min(mask.shape[1], sx + seam_band + 1 - x1)
        if lx0 < lx1 and mask[:, lx0:lx1].any():
            seams.add(("v", sx))

    for sy in seam_ys:
        if sy < y1 - seam_band or sy > y2 + seam_band:
            continue
        ly0 = max(0, sy - seam_band - y1)
        ly1 = min(mask.shape[0], sy + seam_band + 1 - y1)
        if ly0 < ly1 and mask[ly0:ly1, :].any():
            seams.add(("h", sy))

    return seams


def _pair_metrics_on_shared_seams(inst_a, inst_b, shared_seams, seam_band):
    ay1, ay2, ax1, ax2 = inst_a["roi"]
    by1, by2, bx1, bx2 = inst_b["roi"]

    ox1, oy1 = max(ax1, bx1), max(ay1, by1)
    ox2, oy2 = min(ax2, bx2), min(ay2, by2)
    if ox1 >= ox2 or oy1 >= oy2:
        return 0.0, 0.0

    a_sub = inst_a["mask"][oy1 - ay1:oy2 - ay1, ox1 - ax1:ox2 - ax1]
    b_sub = inst_b["mask"][oy1 - by1:oy2 - by1, ox1 - bx1:ox2 - bx1]

    band = np.zeros_like(a_sub, dtype=bool)
    xs = np.arange(ox1, ox2)
    ys = np.arange(oy1, oy2)

    for axis, pos in shared_seams:
        if axis == "v":
            cols = np.abs(xs - pos) <= seam_band
            if np.any(cols):
                band[:, cols] = True
        else:
            rows = np.abs(ys - pos) <= seam_band
            if np.any(rows):
                band[rows, :] = True

    if not band.any():
        return 0.0, 0.0

    a_band = a_sub & band
    b_band = b_sub & band

    inter = np.count_nonzero(a_band & b_band)
    union = np.count_nonzero(a_band | b_band)
    band_iou = inter / union if union > 0 else 0.0

    kernel = np.ones((3, 3), np.uint8)
    a_edge = cv2.morphologyEx(a_band.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    b_edge = cv2.morphologyEx(b_band.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    if not a_edge.any() or not b_edge.any():
        return band_iou, 0.0

    a_edge_d = cv2.dilate(a_edge.astype(np.uint8), kernel, iterations=1) > 0
    touch = np.count_nonzero(a_edge_d & b_edge)
    denom = min(np.count_nonzero(a_edge), np.count_nonzero(b_edge))
    contour_touch = touch / max(1, denom)

    return band_iou, contour_touch


def _boxes_close(box_a, box_b, pad):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 + pad < bx1 or bx2 + pad < ax1 or ay2 + pad < by1 or by2 + pad < ay1)


def _merge_instances(instances, merge_pairs):
    parent = list(range(len(instances)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in merge_pairs:
        union(i, j)

    groups = {}
    for idx in range(len(instances)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    merged = []
    for root in sorted(groups, key=lambda r: min(groups[r])):
        members = groups[root]
        y1 = min(instances[m]["roi"][0] for m in members)
        y2 = max(instances[m]["roi"][1] for m in members)
        x1 = min(instances[m]["roi"][2] for m in members)
        x2 = max(instances[m]["roi"][3] for m in members)

        merged_mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        cls_votes = {}
        best_member = members[0]

        for m in members:
            inst = instances[m]
            iy1, iy2, ix1, ix2 = inst["roi"]
            merged_mask[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] |= inst["mask"]
            cls_votes[inst["cls_id"]] = cls_votes.get(inst["cls_id"], 0) + 1
            if inst["score"] > instances[best_member]["score"]:
                best_member = m

        if not merged_mask.any():
            continue

        cls_id = max(cls_votes.items(), key=lambda kv: kv[1])[0]
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        center = _mask_centroid(merged_mask, x1, y1, box)

        merged.append({
            "cls_id": cls_id,
            "score": instances[best_member]["score"],
            "box": box,
            "roi": (y1, y2, x1, x2),
            "mask": merged_mask,
            "center": center,
        })

    return merged



def run(model_path, image_path, tile_size=640, overlap=100, conf_thres=0.25, iou_thres=0.45,
        enable_seam_merge=True, seam_band=None, seam_iou_thres=0.15,
        contour_touch_thres=0.2, strong_seam_iou_thres=0.28, centroid_dist_thres=None):
    """切片推理 + 掩码渲染，返回 (annotated_bgr, total_count, class_counts)"""
    model = YOLO(model_path)
    img = cv2.imread(image_path)

    keep_idx, all_boxes, all_scores, all_cls, all_masks = _manual_sliced_predict(
        img, model, tile_size, overlap, conf_thres, iou_thres
    )

    if len(keep_idx) == 0:
        return img, 0, {}

    h, w = img.shape[:2]
    stride = tile_size - overlap
    seam_band = seam_band if seam_band is not None else max(2, overlap // 8)
    centroid_dist_thres = centroid_dist_thres if centroid_dist_thres is not None else overlap
    seam_xs = _seam_positions(w, stride)
    seam_ys = _seam_positions(h, stride)

    instances = []
    for i in keep_idx:
        box = all_boxes[i]
        cls_id = int(all_cls[i])
        score = float(all_scores[i])
        m_data, roi = all_masks[i]
        y1, y2, x1, x2 = roi
        center = _mask_centroid(m_data, x1, y1, box)
        seams = _touching_seams(m_data, roi, seam_xs, seam_ys, seam_band)

        instances.append({
            "src_idx": int(i),
            "box": box,
            "cls_id": cls_id,
            "score": score,
            "roi": roi,
            "mask": m_data,
            "center": center,
            "seams": seams,
        })

    if enable_seam_merge and len(instances) > 1:
        merge_pairs = []
        pad = max(4, overlap)
        for i in range(len(instances)):
            a = instances[i]
            if not a["seams"]:
                continue
            for j in range(i + 1, len(instances)):
                b = instances[j]
                if a["cls_id"] != b["cls_id"]:
                    continue
                if not b["seams"]:
                    continue
                shared_seams = a["seams"] & b["seams"]
                if not shared_seams:
                    continue
                if not _boxes_close(a["box"], b["box"], pad=pad):
                    continue

                dx = a["center"][0] - b["center"][0]
                dy = a["center"][1] - b["center"][1]
                if dx * dx + dy * dy > centroid_dist_thres * centroid_dist_thres:
                    continue

                band_iou, contour_touch = _pair_metrics_on_shared_seams(a, b, shared_seams, seam_band)
                if (band_iou >= seam_iou_thres and contour_touch >= contour_touch_thres) or band_iou >= strong_seam_iou_thres:
                    merge_pairs.append((i, j))

        instances = _merge_instances(instances, merge_pairs)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    class_counts = {}

    for count_id, inst in enumerate(instances, start=1):
        cls_id = inst["cls_id"]
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        color = colors[cls_id % len(colors)]

        y1, y2, x1, x2 = inst["roi"]
        m_data = inst["mask"]
        roi = img[y1:y2, x1:x2]
        roi[m_data] = (roi[m_data] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        img[y1:y2, x1:x2] = roi

        cx, cy = inst["center"]
        cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, str(count_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    total = len(instances)
    overlay = img.copy()
    box_h = 70 + len(class_counts) * 35
    cv2.rectangle(overlay, (10, 10), (320, box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, f"TOTAL: {total}", (25, 50), 2, 0.9, (255, 255, 255), 2)
    for idx, (name, count) in enumerate(class_counts.items()):
        cv2.putText(img, f"{name}: {count}", (30, 90 + idx * 35), 2, 0.7, (0, 255, 0), 2)

    return img, total, class_counts


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

    result_img, total, class_counts = run(MODEL_PATH, IMAGE_PATH)
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
