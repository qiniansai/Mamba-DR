import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
from PIL import Image, ImageOps
from tqdm import tqdm

# 配置参数
IMG_SIZE = 384
NUM_WORKERS = 8  # 根据你的CPU核心数调整
FUNDUS_PATH = glob(
    "./woc/*.jpg"  # 替换为你的眼底图像路径
)
SAVE_DIR = "./data/DDR_hhh"  # 处理后眼底图像的保存目录


def pad_to_square(img):
    """
    将图像填充为正方形（短边填充黑色）
    Args:
        img: PIL Image对象
    Returns:
        正方形的PIL Image对象
    """
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        pad = (width - height) // 2
        return ImageOps.expand(img, border=(0, pad, 0, pad), fill=0)
    else:
        pad = (height - width) // 2
        return ImageOps.expand(img, border=(pad, 0, pad, 0), fill=0)


def border_crop_square_padding(img_path: str):
    """
    裁剪图像空白边界 + 填充为正方形（仅处理fundus图像）
    Args:
        img_path: 眼底图像路径
    Returns:
        预处理后的PIL Image对象
    """
    # 读取灰度图并通过Otsu二值化找有效区域边界
    image = cv2.imread(img_path, 0)
    ret2, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 检测有效区域的左边界
    left = 0
    for col in range(thresh.shape[1]):
        if thresh[:, col].sum() > 0:
            left = col
            break
    # 检测有效区域的右边界
    right = 0
    for col in range(thresh.shape[1] - 1, -1, -1):
        if thresh[:, col].sum() > 0:
            right = col
            break
    # 检测有效区域的上边界
    top = 0
    for row in range(thresh.shape[0]):
        if thresh[row, :].sum() > 0:
            top = row
            break
    # 检测有效区域的下边界
    bottom = 0
    for row in range(thresh.shape[0] - 1, -1, -1):
        if thresh[row, :].sum() > 0:
            bottom = row
            break

    # 裁剪到有效区域
    img_cropped = Image.open(img_path)
    img_cropped = img_cropped.crop((left, top, right + 1, bottom + 1))

    # 填充为正方形
    img_cropped = pad_to_square(img_cropped)

    return img_cropped


def process_single_image(fundus_path):
    """
    处理单张眼底图像（仅保留fundus逻辑）
    Args:
        fundus_path: 眼底图像路径
    """
    # 提取文件名（无后缀）
    file_name = os.path.basename(fundus_path).split(".")[0]

    # 预处理：裁剪边界 + 填充正方形
    fundus_img = border_crop_square_padding(fundus_path)

    # 缩放到目标尺寸
    fundus_img = fundus_img.resize((IMG_SIZE, IMG_SIZE))

    # 保存处理后的眼底图像
    fundus_dst = f"{SAVE_DIR}/fundus/{file_name}.jpg"
    os.makedirs(os.path.dirname(fundus_dst), exist_ok=True)
    fundus_img.save(fundus_dst)


if __name__ == "__main__":
    success_cnt, failed_cnt = 0, 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有眼底图像的处理任务
        futures = [
            executor.submit(process_single_image, fundus_path)
            for fundus_path in FUNDUS_PATH
        ]

        # 进度条展示 + 异常捕获
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()
                success_cnt += 1
            except Exception as e:
                failed_cnt += 1
                print(f"处理图像出错: {e}")

    print(f"处理完成 - 成功: {success_cnt}, 失败: {failed_cnt}")