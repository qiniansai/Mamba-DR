"""
Mamba-DR 模型推理脚本
直接从 .pth 权重文件加载 MambaVisionConcept，无需 Lightning。

用法：
  python inference.py --weights model_weights.pth --image path/to/img.jpg [--output ./output]
  python inference.py --weights model_weights.pth --image_dir path/to/images [--output ./output]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

INFERENCE_DIR = Path(__file__).resolve().parent
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from model.encoder.mambavision_concept import MambaVisionConcept  # noqa: E402


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_encoder(weights_path: str, device: torch.device) -> MambaVisionConcept:
    """从 .pth 权重文件加载 MambaVisionConcept。

    .pth 文件结构：{"state_dict": ..., "config": {...}}
    """
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    state = payload["state_dict"]

    encoder = MambaVisionConcept(
        num_lesions=cfg["num_lesions"],
        num_classes=cfg["num_classes"],
        img_size=cfg["img_size"],
        backbone_name=cfg["backbone_name"],
        use_ordinal_head=cfg.get("use_ordinal_head", True),
        ordinal_num_heads=cfg.get("ordinal_num_heads", 8),
        use_level_specific_attention=True,
        lesion_names=cfg["lesion_names"],
    )
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    encoder.to(device)

    # 附加元信息供外部使用
    encoder.disease_names = cfg["disease_names"]
    encoder.lesion_names = cfg["lesion_names"]
    return encoder


# ---------------------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------------------
def build_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(encoder: MambaVisionConcept, image_path: str, device: torch.device):
    """对单张图推理，返回 (raw_img, disease_logits, lesion_logits, cams)。"""
    img_size = encoder.img_size
    transform = build_transform(img_size)
    raw = np.array(Image.open(image_path).convert("RGB"))
    x = transform(image=raw)["image"].unsqueeze(0).to(device)

    output = encoder(x, return_attn=True)
    if isinstance(output, tuple):
        output, _ = output
    return raw, output.disease_logits, output.lesion_logits, output.cams


# ---------------------------------------------------------------------------
# CAM 热力图
# ---------------------------------------------------------------------------
def build_cam_heatmap(raw_img: np.ndarray, cam: np.ndarray, img_size: int) -> np.ndarray:
    """将单概念 CAM 归一化并叠加到原图，返回 RGB 0~1 数组。"""
    cam_np = cv2.resize(cam, (img_size, img_size))
    cmin, cmax = cam_np.min(), cam_np.max()
    if cmax - cmin < 1e-8:
        cam_norm = np.full_like(cam_np, 0.5)
    else:
        cam_norm = (cam_np - cmin) / (cmax - cmin)
    cam_norm = np.clip(np.nan_to_num(cam_norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    raw_rgb = cv2.resize(raw_img, (img_size, img_size))
    heatmap = cv2.applyColorMap((255 * cam_norm).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (0.5 * raw_rgb.astype(np.float32) + 0.5 * heatmap.astype(np.float32)) / 255.0


def save_cam_visualizations(
    encoder: MambaVisionConcept,
    raw_img: np.ndarray,
    cams: torch.Tensor,
    lesion_logits: torch.Tensor,
    disease_logits: torch.Tensor,
    output_dir: str,
    stem: str,
    img_size: int,
) -> list:
    out_dir = os.path.join(output_dir, "cam")
    os.makedirs(out_dir, exist_ok=True)
    lesion_names = encoder.lesion_names
    n = cams.shape[1]
    lconf = torch.sigmoid(lesion_logits)[0].detach().cpu().numpy()
    dclass = int(disease_logits[0].argmax().item())
    cams_r = (
        F.interpolate(cams, size=(img_size, img_size), mode="bilinear", align_corners=False)
        .detach().cpu().numpy()
    )
    saved = []
    for i in range(n):
        name = lesion_names[i] if i < len(lesion_names) else f"concept_{i}"
        overlay = (build_cam_heatmap(raw_img, cams_r[0, i], img_size) * 255).astype(np.uint8)
        fname = f"{stem}__{name}__score_{lconf[i]:.4f}__ds_{dclass}.png"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# 打印结果
# ---------------------------------------------------------------------------
def print_predictions(encoder, image_path, disease_logits, lesion_logits):
    dp = torch.softmax(disease_logits[0], dim=0).detach().cpu().numpy()
    di = int(dp.argmax())
    lconf = torch.sigmoid(lesion_logits[0]).detach().cpu().numpy()
    print(f"\n===== {image_path} =====")
    print("疾病分级：")
    for i, name in enumerate(encoder.disease_names):
        print(f"  {name}: {dp[i]:.4f}{'  <==' if i == di else ''}")
    print("病灶：")
    for i, name in enumerate(encoder.lesion_names):
        print(f"  {name}: {lconf[i]:.4f}")


def collect_image_paths(input_dir: str) -> list:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(exts)
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mamba-DR 推理 + CAM 热力图")
    parser.add_argument("--weights", type=str, required=True, help="model_weights.pth 路径")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, default=None, help="图片目录路径")
    parser.add_argument("--output", type=str, default="./output", help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    args = parser.parse_args()

    if (args.image is None) == (args.image_dir is None):
        parser.error("必须且只能指定 --image 或 --image_dir 之一")

    # 显式指定则尊重；否则自动选择（CUDA 可用优先）
    if args.device in ("cuda", "cpu"):
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    encoder = load_encoder(args.weights, device)
    img_size = encoder.img_size
    os.makedirs(args.output, exist_ok=True)

    image_paths = [args.image] if args.image else collect_image_paths(args.image_dir)
    print(f"共 {len(image_paths)} 张图片待推理...")

    for img_path in image_paths:
        raw, dl, ll, cams = predict(encoder, img_path, device)
        print_predictions(encoder, img_path, dl, ll)
        if cams is not None:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            saved = save_cam_visualizations(
                encoder, raw, cams, ll, dl, args.output, stem, img_size
            )
            print(f"  CAM 热力图已保存 {len(saved)} 张到 {os.path.dirname(saved[0])}")

    print("\n推理完成。")


if __name__ == "__main__":
    main()
