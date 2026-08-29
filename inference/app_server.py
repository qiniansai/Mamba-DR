"""
Mamba-DR Flask 推理 API 服务

  GET  /health          - 健康检查
  POST /predict         - 上传单张眼底图，返回疾病分级 + 病灶概率；可选生成 CAM 热力图(base64)

用法：
  python app_server.py --weights model_weights.pth [--host 0.0.0.0] [--port 5000]

请求示例：
  curl -X POST -F "file=@1.jpg" -F "cam=true" http://127.0.0.1:5000/predict

可通过环境变量 MAMBA_DR_WEIGHTS 指定权重文件路径（优先级低于 --weights 参数）。
"""

import argparse
import base64
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from inference import load_encoder, build_transform, build_cam_heatmap  # noqa: E402

from flask import Flask, jsonify, request  # noqa: E402

app = Flask(__name__)

_ENCODER = None
_DEVICE = None


def get_encoder(weights_path: str):
    """懒加载模型（进程内缓存）。"""
    global _ENCODER, _DEVICE
    if _ENCODER is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"加载模型: {weights_path} -> {_DEVICE}")
        _ENCODER = load_encoder(weights_path, _DEVICE)
    return _ENCODER, _DEVICE


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _ENCODER is not None})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """单张眼底图推理。

    Multipart 参数：file=图片文件（必填）
    Query 参数：    cam=true/false 是否返回 CAM 热力图 base64（默认 false）
    """
    from io import BytesIO

    weights = os.environ.get("MAMBA_DR_WEIGHTS") or os.path.join(INFERENCE_DIR, "model_weights.pth")
    if not os.path.exists(weights):
        return jsonify({"error": f"权重文件不存在: {weights}"}), 400

    if "file" not in request.files:
        return jsonify({"error": "缺少 'file' 字段，请上传图片"}), 400

    want_cam = request.args.get("cam", "false").lower() in ("true", "1", "yes")

    try:
        img_bytes = request.files["file"].read()
        raw_img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"图片解码失败: {str(e)}"}), 400

    encoder, device = get_encoder(weights)
    img_size = encoder.img_size
    transform = build_transform(img_size)

    try:
        x = transform(image=raw_img)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = encoder(x, return_attn=True)
            if isinstance(output, tuple):
                output, _ = output
            disease_logits = output.disease_logits
            lesion_logits = output.lesion_logits
            cams = output.cams

        disease_probs = torch.softmax(disease_logits[0], dim=0).detach().cpu().numpy()
        disease_idx = int(disease_probs.argmax())
        lesion_conf = torch.sigmoid(lesion_logits[0]).detach().cpu().numpy()

        result = {
            "disease_pred": int(disease_idx),
            "disease": {name: round(float(disease_probs[i]), 6)
                        for i, name in enumerate(encoder.disease_names)},
            "lesions": {name: round(float(lesion_conf[i]), 6)
                        for i, name in enumerate(encoder.lesion_names)},
        }

        if want_cam and cams is not None:
            cams_resized = F.interpolate(
                cams, size=(img_size, img_size), mode="bilinear", align_corners=False,
            ).detach().cpu().numpy()

            heatmaps = {}
            for concept_ind in range(cams.shape[1]):
                concept_name = (
                    encoder.lesion_names[concept_ind]
                    if concept_ind < len(encoder.lesion_names) else f"concept_{concept_ind}"
                )
                overlay = build_cam_heatmap(raw_img, cams_resized[0, concept_ind], img_size)
                overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", overlay_bgr)
                if ok:
                    heatmaps[concept_name] = base64.b64encode(buf.tobytes()).decode("ascii")

            result["heatmaps"] = heatmaps

        return jsonify(result)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"推理失败: {str(e)}"}), 500


def main():
    parser = argparse.ArgumentParser(description="Mamba-DR Flask 推理 API")
    parser.add_argument("--weights", type=str, default=None, help="model_weights.pth 路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    weights = args.weights or os.environ.get("MAMBA_DR_WEIGHTS") or os.path.join(
        INFERENCE_DIR, "model_weights.pth")
    if not os.path.exists(weights):
        print(f"警告: 未找到权重文件 {weights}，请用 --weights 指定")
    else:
        get_encoder(weights)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
