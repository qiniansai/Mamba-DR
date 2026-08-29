import os
import shutil
from pathlib import Path

def find_and_copy_unique_images():
    # ===================== 配置参数 =====================
    # 请根据你的实际路径修改以下变量
    folder_a = "./A"    # 文件夹A路径
    folder_b = "./B"    # 文件夹B路径
    folder_c = "./C"    # 源文件所在文件夹C
    folder_d = "./D"    # 目标文件夹D
    # 定义需要处理的图片格式（可根据需要添加/删除）
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    # ===================== 步骤1：读取文件夹文件名 =====================
    def get_image_filenames(folder_path):
        """读取指定文件夹中的所有图片文件名（仅返回文件名，不含路径）"""
        filenames = []
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 {folder_path} 不存在！")
            return filenames
        
        for file in os.listdir(folder_path):
            # 过滤文件夹，只保留文件；过滤非图片格式
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(image_extensions):
                filenames.append(file)
        return filenames

    # 读取A、B文件夹的图片文件名
    files_a = get_image_filenames(folder_a)
    files_b = get_image_filenames(folder_b)
    
    if not files_a:
        print("错误：文件夹A中未找到任何图片文件！")
        return
    if not files_b:
        print("警告：文件夹B中未找到任何图片文件，将处理文件夹A的所有文件")

    # ===================== 步骤2：计算A独有的文件 =====================
    # 转换为集合方便求差集
    set_a = set(files_a)
    set_b = set(files_b)
    # A中有但B中没有的文件
    unique_files = list(set_a - set_b)
    
    print(f"文件夹A中的图片数量：{len(files_a)}")
    print(f"文件夹B中的图片数量：{len(files_b)}")
    print(f"文件夹A独有的图片数量：{len(unique_files)}")
    
    if not unique_files:
        print("提示：文件夹A中没有独有的图片文件，无需复制")
        return

    # 可选：打印前10个独有文件（方便验证）
    print("\n前10个A独有的文件：")
    for i, file in enumerate(unique_files[:10], 1):
        print(f"  {i}. {file}")
    if len(unique_files) > 10:
        print(f"  ... 还有 {len(unique_files)-10} 个文件")

    # ===================== 步骤3：复制文件到D =====================
    # 创建目标文件夹D（如果不存在）
    Path(folder_d).mkdir(parents=True, exist_ok=True)
    
    copied_count = 0    # 成功复制的文件数
    missing_count = 0   # C中不存在的文件数
    error_count = 0     # 复制失败的文件数

    print(f"\n开始从文件夹C复制文件到文件夹D...")
    for filename in unique_files:
        # 源文件路径（C文件夹中）
        source_path = os.path.join(folder_c, filename)
        # 目标文件路径（D文件夹中）
        target_path = os.path.join(folder_d, filename)

        try:
            # 检查C中是否存在该文件
            if not os.path.exists(source_path):
                print(f"警告：文件夹C中未找到文件 {filename}，跳过")
                missing_count += 1
                continue
            
            # 复制文件（copy2会保留文件元数据，如创建时间）
            shutil.copy2(source_path, target_path)
            copied_count += 1
            
        except Exception as e:
            # 捕获复制过程中的异常（如权限不足、文件损坏等）
            print(f"错误：复制 {filename} 失败，原因：{str(e)}")
            error_count += 1

    # ===================== 输出结果统计 =====================
    print("\n" + "="*50)
    print("复制完成！统计结果：")
    print(f"✅ 成功复制文件数：{copied_count}")
    print(f"❌ 文件夹C中缺失的文件数：{missing_count}")
    print(f"⚠️  复制失败的文件数：{error_count}")
    print(f"📁 目标文件夹路径：{os.path.abspath(folder_d)}")

if __name__ == "__main__":
    find_and_copy_unique_images()