# spectro_gradcam_visualizer.py
# GradCAM 热力图生成器 + 支持 JSON 坐标轴信息叠加

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from PIL import Image

def generate_gradcam_overlay(
    model, image_path, json_path,
    target_class_idx, target_layer_name,
    output_path, return_cam=False
):
    model.eval()
    device = next(model.parameters()).device

    # 读取图像 + 转换
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 读取频谱图元数据
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    meta_block = metadata.get('cqt', metadata.get('cwt', metadata.get('scalogram', {})))
    freq_min = meta_block.get('freq_min', 0)
    freq_max = meta_block.get('freq_max', 8000)
    duration = metadata.get('duration_sec', 2.0)

    # 提取层并挂 hook
    fmap, grad = {}, {}
    layer = dict(model.named_modules())[target_layer_name]
    def fwd(m, i, o): fmap['val'] = o.detach()
    def bwd(m, gi, go): grad['val'] = go[0].detach()
    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_backward_hook(bwd)

    output, _ = model(input_tensor)
    score = output[0, target_class_idx]
    model.zero_grad()
    score.backward()
    h1.remove(); h2.remove()

    # 计算 CAM 热力图
    feat, grads = fmap['val'], grad['val']
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * feat).sum(dim=1)[0].cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.width, image.height))
    cam -= cam.min(); cam /= cam.max(); cam = (cam * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

    # 绘图：添加频率时间刻度
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(overlay)
    ax.set_title("GradCAM Overlay")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xticks([0, image.width])
    ax.set_xticklabels(["0", f"{duration:.1f}"])
    ax.set_yticks([0, image.height])
    ax.set_yticklabels([f"{freq_max:.0f}", f"{freq_min:.0f}"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved GradCAM overlay to {output_path}")

    if return_cam:
        return cam
