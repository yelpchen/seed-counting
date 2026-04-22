import os
import cv2


def slice_image_only(image_path, output_dir, slice_size=640, overlap_ratio=0.2):
    """
    仅切割图片为固定大小的切片（带重叠），不处理任何标注
    """
    # 创建输出目录
    img_output_dir = os.path.join(output_dir, "images")
    os.makedirs(img_output_dir, exist_ok=True)

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]  # 自动获取尺寸
    print(f"📏 原图尺寸: {w} x {h}")

    # 计算步长
    step = int(slice_size * (1 - overlap_ratio))
    step = max(1, step)  # 防止为0

    slice_count = 0

    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + slice_size, w)
            y_end = min(y + slice_size, h)
            slice_img = img[y:y_end, x:x_end]

            # 如果切片小于 640x640，则填充黑边
            if slice_img.shape[0] < slice_size or slice_img.shape[1] < slice_size:
                pad_bottom = slice_size - slice_img.shape[0]
                pad_right = slice_size - slice_img.shape[1]
                slice_img = cv2.copyMakeBorder(
                    slice_img,
                    0, pad_bottom,
                    0, pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)  # 黑色填充
                )

            # 保存切片
            slice_filename = f"slice_{slice_count:04d}.png"
            save_path = os.path.join(img_output_dir, slice_filename)
            cv2.imwrite(save_path, slice_img)

            slice_count += 1

    print(f"✅ 切割完成！共生成 {slice_count} 个 640x640 切片（含重叠）。")
    print(f"📁 保存位置: {img_output_dir}")


if __name__ == "__main__":
    # 配置参数
    image_path = "image/3.png"  # 输入图片
    output_dir = "image/3"  # 输出根目录
    slice_size = 640  # 切片大小
    overlap_ratio = 0.2  # 重叠比例 20%

    slice_image_only(image_path, output_dir, slice_size, overlap_ratio)