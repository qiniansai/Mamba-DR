import os
import random
import shutil
from pathlib import Path

def copy_selected_images():
    # 配置参数
    txt_file_path = "train.txt"          # 标签文件路径
    source_folder = "train"             # 源图片文件夹
    target_folder = "selected_images"   # 目标文件夹（复制到这里）
    target_label = 0                    # 要筛选的标签
    sample_count = 743                  # 要抽取的数量

    # 创建目标文件夹（如果不存在）
    Path(target_folder).mkdir(exist_ok=True)

    # 1. 读取并筛选文件
    label_0_files = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 去除首尾空白字符
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 分割每行内容（按空格分割，只分割一次，避免文件名含空格）
                    parts = line.split(' ', 1)
                    filename = parts[0]
                    label = int(parts[1])
                    
                    # 筛选标签为0的文件
                    if label == target_label:
                        label_0_files.append(filename)
                except (IndexError, ValueError) as e:
                    print(f"警告：第{line_num}行格式错误，已跳过。错误信息：{e}")
                    continue
    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_file_path}，请检查文件路径是否正确！")
        return
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return

    # 2. 检查可用文件数量
    available_count = len(label_0_files)
    print(f"找到标签为{target_label}的文件数量：{available_count}")
    
    if available_count < sample_count:
        print(f"错误：可用文件数量({available_count})少于需要抽取的数量({sample_count})！")
        return

    # 3. 随机抽取指定数量的文件
    random.seed(42)  # 设置随机种子，确保结果可复现
    selected_files = random.sample(label_0_files, sample_count)
    print(f"已随机抽取 {len(selected_files)} 个文件")

    # 4. 复制文件
    copied_count = 0
    for filename in selected_files:
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        
        try:
            # 检查源文件是否存在
            if not os.path.exists(source_path):
                print(f"警告：源文件不存在，跳过：{source_path}")
                continue
            
            # 复制文件（覆盖已存在的文件）
            shutil.copy2(source_path, target_path)
            copied_count += 1
        except Exception as e:
            print(f"复制文件时出错 {filename}：{e}")
            continue

    # 5. 输出结果
    print(f"\n复制完成！")
    print(f"成功复制：{copied_count} 个文件")
    print(f"目标文件夹：{os.path.abspath(target_folder)}")

if __name__ == "__main__":
    copy_selected_images()