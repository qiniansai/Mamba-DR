import os
import csv
from pathlib import Path

def generate_disease_grade_csv():
    # ===================== 配置参数 =====================
    # 请根据你的实际路径修改以下变量
    folder_a = "./DDR/fundus_384"       # 文件夹A路径（存放图片）
    txt_file = "total.txt"              # 疾病分级文本文件路径
    csv_output = "disease_grades.csv"   # 输出CSV文件路径
    # 支持的图片格式（可根据需要添加）
    image_extensions = ('.jpg', '.jpeg')

    # ===================== 步骤1：读取文件夹A的图片文件名 =====================
    def get_image_filenames(folder_path):
        """读取文件夹中的所有图片文件名（保留后缀）"""
        filenames = []
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 {folder_path} 不存在！")
            return filenames
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # 过滤文件类型：仅保留图片，且后缀为jpg/jpeg
            if os.path.isfile(file_path) and file.lower().endswith(image_extensions):
                filenames.append(file)
        
        if not filenames:
            print(f"警告：文件夹 {folder_path} 中未找到任何jpg/jpeg图片文件！")
        else:
            print(f"成功读取文件夹A的图片数量：{len(filenames)}")
        return filenames

    # 获取文件夹A的图片列表
    a_images = get_image_filenames(folder_a)
    if not a_images:
        return  # 无图片则终止程序

    # ===================== 步骤2：解析total.txt，建立文件名-分级映射 =====================
    grade_mapping = {}  # 格式：{"12345.jpg": 0, ...}
    error_lines = []    # 记录格式错误的行

    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 分割每行内容（按空格分割1次，避免文件名含空格）
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    error_lines.append(f"第{line_num}行：格式错误（未找到文件名和分级）")
                    continue
                
                filename = parts[0].strip()
                grade_str = parts[1].strip()

                # 验证分级是否为数字
                try:
                    grade = int(grade_str)
                    # 检查分级是否合法（0-4为有效，5报错，其他无效）
                    if grade < 0 or grade > 5:
                        error_lines.append(f"第{line_num}行：分级{grade}无效（仅支持0-5）")
                        continue
                    grade_mapping[filename] = grade
                except ValueError:
                    error_lines.append(f"第{line_num}行：分级{grade_str}不是有效数字")
                    continue
    except FileNotFoundError:
        print(f"错误：未找到文件 {txt_file}，请检查路径是否正确！")
        return
    except Exception as e:
        print(f"读取total.txt时发生错误：{str(e)}")
        return

    # 输出txt解析结果
    print(f"成功解析total.txt的有效分级记录数：{len(grade_mapping)}")
    if error_lines:
        print(f"\n⚠️  total.txt格式错误的行（共{len(error_lines)}行）：")
        for err in error_lines[:10]:  # 只显示前10条错误
            print(f"  {err}")
        if len(error_lines) > 10:
            print(f"  ... 还有 {len(error_lines)-10} 条错误行")

    # ===================== 步骤3：生成CSV文件 =====================
    # 定义CSV表头
    csv_headers = ["ID", "No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    # 存储CSV行数据
    csv_rows = []

    # 遍历文件夹A的每个图片文件
    missing_files = []  # 记录A中在total.txt中未找到的文件
    for img_file in a_images:
        # 检查该图片是否在分级映射中
        if img_file not in grade_mapping:
            missing_files.append(img_file)
            continue
        
        grade = grade_mapping[img_file]
        
        # 分级为5则立即报错并终止程序
        if grade == 5:
            print(f"\n❌ 错误：图片 {img_file} 的疾病分级为5，不符合要求，终止程序！")
            return
        
        # 提取ID（去除.jpg/.jpeg后缀）
        img_name = Path(img_file).stem  # 自动处理不同后缀，如12345.jpg → 12345
        
        # 初始化所有列值为0
        row_data = [0] * len(csv_headers)
        row_data[0] = img_name  # ID列赋值
        
        # 根据分级设置对应列的值为1
        if grade == 0:
            row_data[1] = 1  # No DR列
        elif grade == 1:
            row_data[2] = 1  # Mild列
        elif grade == 2:
            row_data[3] = 1  # Moderate列
        elif grade == 3:
            row_data[4] = 1  # Severe列
        elif grade == 4:
            row_data[5] = 1  # Proliferative列
        
        csv_rows.append(row_data)

    # ===================== 步骤4：写入CSV文件 =====================
    try:
        with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)  # 写入表头
            writer.writerows(csv_rows)    # 写入所有行数据
        print(f"\n✅ CSV文件生成成功！路径：{os.path.abspath(csv_output)}")
        print(f"📊 CSV文件包含的记录数：{len(csv_rows)}")
    except Exception as e:
        print(f"\n❌ 写入CSV文件失败：{str(e)}")
        return

    # 输出未找到分级的文件（可选）
    if missing_files:
        print(f"\n⚠️  文件夹A中有{len(missing_files)}个图片在total.txt中未找到分级：")
        for f in missing_files[:10]:  # 只显示前10个
            print(f"  {f}")
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files)-10} 个文件未找到分级")

if __name__ == "__main__":
    generate_disease_grade_csv()