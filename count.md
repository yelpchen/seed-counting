# 种子小目标切片拼接线重复计数改进方法（已实现）

## 1. 问题背景

切片推理（`tile_size` + `overlap`）会在拼接线附近产生重复检测：同一粒种子可能在相邻切片中各出现一次。仅用框级 NMS（`batched_nms`）时，若两个框的 IoU 不高，重复目标可能保留，导致计数偏大。

本次实现改为：

- 先做框级 NMS；
- 再做“拼接线感知（seam-aware）”的掩码级合并；
- 最终按合并后的实例渲染与计数。

---

## 2. 实现位置

核心实现文件：`main-workshop/mymask.py`

新增/改造的关键函数：

1. `_seam_positions(length, stride)`
2. `_mask_centroid(mask, x1, y1, box)`
3. `_touching_seams(mask, roi, seam_xs, seam_ys, seam_band)`
4. `_pair_metrics_on_shared_seams(inst_a, inst_b, shared_seams, seam_band)`
5. `_boxes_close(box_a, box_b, pad)`
6. `_merge_instances(instances, merge_pairs)`
7. `run(...)`（主流程接入 seam-aware 合并）

同时兼容了返回值：`run()` 统一返回 `(annotated_bgr, total_count, class_counts)`。

---

## 3. 方法原理

## 3.1 生成拼接线

在 `run()` 中计算：

- `stride = tile_size - overlap`
- 垂直拼接线集合 `seam_xs = _seam_positions(w, stride)`
- 水平拼接线集合 `seam_ys = _seam_positions(h, stride)`

并定义拼接线带宽：

- `seam_band = max(2, overlap // 8)`（默认）

用于只关注拼接线附近区域，而不是全图盲目比较。

---

## 3.2 为每个实例打“触线标签”

对 `keep_idx` 保留实例，记录：

- 类别 `cls_id`
- 置信度 `score`
- ROI `(y1, y2, x1, x2)`
- 局部掩码 `mask`
- 质心 `center`
- 接触到的拼接线集合 `seams`

其中 `seams` 由 `_touching_seams()` 计算：若掩码在某条拼接线 ± `seam_band` 的窄带内有像素，则标记为接触该线。

---

## 3.3 候选对筛选（减少误并和计算量）

只比较满足以下条件的实例对 `(A, B)`：

1. 同类别（`cls_id` 一致）
2. 两者都接触拼接线
3. 有共享拼接线（`A.seams ∩ B.seams` 非空）
4. 框邻近（`_boxes_close`）
5. 质心距离不超过阈值（默认 `centroid_dist_thres = overlap`）

---

## 3.4 拼接线邻域相似度判定

在共享拼接线的窄带内，计算两类指标（`_pair_metrics_on_shared_seams`）：

1. **band_iou**：拼接线窄带内的掩码 IoU
2. **contour_touch**：窄带内轮廓接触比例（轮廓梯度 + 1 次膨胀）

合并规则：

- `(band_iou >= seam_iou_thres 且 contour_touch >= contour_touch_thres)`
  **或**
- `band_iou >= strong_seam_iou_thres`

默认阈值：

- `seam_iou_thres=0.15`
- `contour_touch_thres=0.2`
- `strong_seam_iou_thres=0.28`

---

## 3.5 并查集合并（Union-Find）

对所有满足条件的 pair 用并查集聚类，再对每个簇：

- 进行掩码 OR 合并（得到 `merged_mask`）
- 类别用多数投票
- 分数取簇内最高分实例
- ROI 取簇内最小包围区域
- 中心点由合并掩码质心计算

最终渲染和计数基于合并后的实例列表，避免同一粒种子在拼接线两侧重复计数。

---

## 4. 改进后完整流程（切片模式）

1. 切片推理收集框、分数、类别、掩码
2. 框级 `batched_nms`
3. 构建实例元信息（含触线标签）
4. seam-aware 候选筛选
5. 拼接线邻域指标判定
6. 并查集合并实例
7. 基于合并结果渲染掩码/编号并统计类别数量

---

## 5. 关键参数说明

`run()` 新增参数：

- `enable_seam_merge=True`：是否启用拼接线合并
- `seam_band=None`：拼接线带宽，默认 `max(2, overlap // 8)`
- `seam_iou_thres=0.15`
- `contour_touch_thres=0.2`
- `strong_seam_iou_thres=0.28`
- `centroid_dist_thres=None`：默认等于 `overlap`

---

## 6. 调参建议

- 重复计数仍偏多：
  - 适当降低 `seam_iou_thres`（如 0.15 → 0.12）
  - 适当降低 `strong_seam_iou_thres`（如 0.28 → 0.24）
- 误合并偏多（两粒靠近被并成一粒）：
  - 提高 `contour_touch_thres`（如 0.2 → 0.3）
  - 降低 `centroid_dist_thres`（如 `overlap` → `0.8*overlap`）

建议优先在“拼接线附近样本”上做小步调参，再全量验证。

---

## 7. 当前实现结论

该实现不是全局任意实例合并，而是“**仅在共享拼接线附近**”进行掩码级判定与合并，目标明确、代价可控，能够显著降低切片边界导致的重复计数，同时尽量减少对非边界区域的误伤。