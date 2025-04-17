
import os
import cv2
import json
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============ 频谱图计算与保存 ============

def compute_cqt_spectrogram(y, sr):
    cqt = librosa.cqt(y, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    freqs = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C1'))
    return cqt_db, freqs

def compute_cwt_scalogram(y, sr):
    import scipy.signal
    widths = np.arange(1, 128)
    cwtmatr, freqs = scipy.signal.cwt(y, scipy.signal.ricker, widths), widths
    cwt_db = librosa.amplitude_to_db(np.abs(cwtmatr), ref=np.max)
    return cwt_db, freqs

def save_spectrogram_image(spec_db, cmap, output_path):
    norm_img = cv2.normalize(spec_db, None, 0, 255, cv2.NORM_MINMAX)
    norm_img = norm_img.astype(np.uint8)
    color_img = cv2.applyColorMap(norm_img, cmap)
    cv2.imwrite(output_path, color_img)

# ============ GradCAM 可视化 ============

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def generate_gradcam(model, input_image_path, target_layer_name, target_class_idx, output_path):
    model.eval()
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    target_layer = dict(model.named_modules()).get(target_layer_name, None)
    if target_layer is None:
        raise ValueError(f"Layer {target_layer_name} not found in model.")
    activation, gradient = {}, {}

    def forward_hook(module, inp, out): activation[target_layer_name] = out.detach()
    def backward_hook(module, grad_in, grad_out): gradient[target_layer_name] = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    score = output[0, target_class_idx]
    model.zero_grad()
    score.backward()
    handle_f.remove()
    handle_b.remove()

    act, grad = activation[target_layer_name], gradient[target_layer_name]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1)).squeeze().cpu().numpy()
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    cv2.imwrite(output_path, overlay)
    print(f"Grad-CAM saved: {output_path}")
