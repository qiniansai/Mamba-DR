import os
import csv
from pathlib import Path

def generate_image_existence_csv():
    # ===================== 配置参数 =====================
    # 请根据你的实际路径修改以下变量
    folder_a = "./DDR/fundus_384"       # 源图片文件夹A路径
    folder_b = "./DDR/mask"       # 包含子文件夹的B文件夹路径
    csv_output = "lesion.csv"  # 输出CSV文件路径
    # 要检查的子文件夹列表（固定为EX/HE/MA/SE）
    subfolders = ["EX", "HE", "MA", "SE"]
    # 支持的图片格式（仅匹配.jpg，可根据需要扩展）
    image_ext = ".jpg"

    # ===================== 步骤1：读取文件夹A的图片文件名 =====================
    def get_a_images(folder_path):
        """读取文件夹A的所有.jpg图片文件名（带后缀），返回列表"""
        a_images = []
        if not os.path.exists(folder_path):
            print(f"错误：文件夹A {folder_path} 不存在！")
            return a_images
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # 仅保留文件且后缀为.jpg（不区分大小写）
            if os.path.isfile(file_path) and file.lower().endswith(image_ext):
                a_images.append(file)
        
        if not a_images:
            print(f"警告：文件夹A中未找到任何{image_ext}格式的图片！")
        else:
            print(f"成功读取文件夹A的图片数量：{len(a_images)}")
        return a_images

    a_images = get_a_images(folder_a)
    if not a_images:
        return  # 无图片则终止程序

    # ===================== 步骤2：预加载B文件夹子目录的文件名（快速查询） =====================
    # 构建字典：key=子文件夹名，value=该文件夹下的文件名集合（方便快速查询）
    subfolder_files = {}
    missing_subfolders = []  # 记录B中不存在的子文件夹

    for sub in subfolders:
        sub_path = os.path.join(folder_b, sub)
        if not os.path.exists(sub_path):
            missing_subfolders.append(sub)
            subfolder_files[sub] = set()  # 不存在则设为空集合
            continue
        
        # 读取该子文件夹的所有.jpg文件名，存入集合
        files = set()
        for file in os.listdir(sub_path):
            file_path = os.path.join(sub_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(image_ext):
                files.add(file)
        subfolder_files[sub] = files
        print(f"读取文件夹B/{sub}的图片数量：{len(files)}")

    # 提示缺失的子文件夹
    if missing_subfolders:
        print(f"\n警告：文件夹B中未找到以下子文件夹：{', '.join(missing_subfolders)}")

    # ===================== 步骤3：检查每个图片的存在性并生成CSV数据 =====================
    # 定义CSV表头
    csv_headers = ["ID"] + subfolders
    # 存储CSV行数据
    csv_rows = []

    for img_file in a_images:
        # 提取ID（去除.jpg后缀）
        img_id = Path(img_file).stem  # 自动处理：12345.jpg → 12345
        
        # 初始化行数据：ID + 四个子文件夹的默认值0
        row_data = [img_id] + [0] * len(subfolders)
        
        # 依次检查每个子文件夹
        for idx, sub in enumerate(subfolders):
            # 子文件夹索引从1开始（0是ID列）
            if img_file in subfolder_files[sub]:
                row_data[idx + 1] = 1
        
        csv_rows.append(row_data)

    # ===================== 步骤4：写入CSV文件 =====================
    try:
        with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)  # 写入表头
            writer.writerows(csv_rows)    # 写入所有行数据
        
        print(f"\n✅ CSV文件生成成功！")
        print(f"📁 文件路径：{os.path.abspath(csv_output)}")
        print(f"📊 包含记录数：{len(csv_rows)}")
    except PermissionError:
        print(f"\n❌ 写入CSV失败：权限不足，请关闭已打开的{csv_output}文件后重试")
    except Exception as e:
        print(f"\n❌ 写入CSV文件失败：{str(e)}")
        return

    # ===================== 可选：统计各列的1的数量 =====================
    print("\n📈 各子文件夹匹配统计：")
    for idx, sub in enumerate(subfolders):
        count = sum([row[idx+1] for row in csv_rows])
        print(f"  {sub}：{count} 个文件匹配")

if __name__ == "__main__":
    generate_image_existence_csv()