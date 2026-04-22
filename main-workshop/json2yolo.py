import json
import os
import cv2
import argparse
from tqdm import tqdm


def convert_labelme_to_yolo(json_path, output_dir, class_names):
    """
    将单个LabelMe JSON文件转换为YOLO格式

    参数:
        json_path: LabelMe标注的JSON文件路径
        output_dir: 输出YOLO格式标签的目录
        class_names: 类别名称列表，用于将类别名称映射为索引
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图像宽高
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 获取输出文件名（与图片名相同，后缀改为txt）
    img_name = os.path.splitext(os.path.basename(data['imagePath']))[0]
    txt_path = os.path.join(output_dir, f"{img_name}.txt")

    # 处理每个标注
    with open(txt_path, 'w', encoding='utf-8') as f:
        for shape in data['shapes']:
            # 获取类别名称并转换为索引
            label = shape['label']
            if label not in class_names:
                print(f"警告: 类别 '{label}' 不在类别列表中，已跳过")
                continue
            class_id = class_names.index(label)

            # 获取标注点并归一化
            points = shape['points']
            normalized_points = []

            for (x, y) in points:
                # 归一化到[0,1]范围
                norm_x = x / img_width
                norm_y = y / img_height
                normalized_points.append(f"{norm_x:.6f} {norm_y:.6f}")

            # 对于多边形，YOLO格式为: 类别ID x1 y1 x2 y2 ... xn yn
            line = f"{class_id} " + " ".join(normalized_points) + "\n"
            f.write(line)

    return txt_path


def batch_convert(input_dir, output_dir, class_names_path):
    """
    批量转换目录中的所有LabelMe JSON文件

    参数:
        input_dir: 包含LabelMe JSON文件的目录
        output_dir: 输出YOLO格式标签的目录
        class_names_path: 类别名称文件路径
    """
    # 读取类别名称
    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if not json_files:
        print(f"警告: 在目录 '{input_dir}' 中未找到任何JSON文件")
        return

    # 批量转换
    print(f"找到 {len(json_files)} 个JSON文件，开始转换...")
    for json_file in tqdm(json_files):
        json_path = os.path.join(input_dir, json_file)
        try:
            convert_labelme_to_yolo(json_path, output_dir, class_names)
        except Exception as e:
            print(f"转换文件 '{json_file}' 时出错: {str(e)}")

    print(f"转换完成，结果保存在: {output_dir}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将LabelMe标注的JSON文件转换为YOLO格式')
    parser.add_argument('--input', type=str, required=True, help='输入JSON文件或包含JSON文件的目录')
    parser.add_argument('--output', type=str, required=True, help='输出YOLO格式标签的目录')
    parser.add_argument('--classes', type=str, required=True, help='包含类别名称的文本文件路径')

    args = parser.parse_args()

    # 检查输入是文件还是目录
    if os.path.isfile(args.input) and args.input.endswith('.json'):
        # 单个文件转换
        with open(args.classes, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
        convert_labelme_to_yolo(args.input, args.output, class_names)
        print(f"转换完成，结果保存在: {args.output}")
    elif os.path.isdir(args.input):
        # 批量转换
        batch_convert(args.input, args.output, args.classes)
    else:
        print(f"错误: 输入 '{args.input}' 不是有效的JSON文件或目录")

