import os


def batch_rename(folder_path, prefix):
    """
    在指定文件夹下的所有文件名前添加前缀
    """
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"❌ 错误：路径 '{folder_path}' 不存在")
        return

    # 获取文件夹内所有文件
    files = os.listdir(folder_path)

    count = 0
    print(f"🚀 开始处理：{folder_path}")

    for filename in files:
        # 构建完整路径
        old_path = os.path.join(folder_path, filename)

        # 确保只处理文件，跳过子文件夹
        if os.path.isfile(old_path):
            # 检查是否已经包含该前缀，避免重复添加
            if filename.startswith(prefix):
                print(f"⏭️ 跳过：{filename} (已包含前缀)")
                continue

            # 生成新文件名
            new_filename = prefix + filename
            new_path = os.path.join(folder_path, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"✅ 已重命名: {filename} -> {new_filename}")
                count += 1
            except Exception as e:
                print(f"❌ 重命名 {filename} 失败: {e}")

    print(f"\n✨ 处理完成！共重命名了 {count} 个文件。")


# --- 设置区 ---
if __name__ == "__main__":
    # 1. 在这里输入你的文件夹路径（建议使用绝对路径）
    # 注意：Windows 路径建议在引号前加 r 防止转义，例如 r"C:\Desktop\Photos"
    target_folder = r"./your_folder_path"

    # 2. 在这里输入你想要添加的自定义文字
    custom_prefix = "【工作资料】_"

    # 执行
    batch_rename(target_folder, custom_prefix)