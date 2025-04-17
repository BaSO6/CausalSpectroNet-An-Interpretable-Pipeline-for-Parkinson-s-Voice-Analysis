# ✅ GradCAM 可解释性分析 - 用于 CausalSpectroNet（ConstantQOnly 模型）
# 请确保 `spectro_ablation_experiments.py` 与 `.pth` 模型位于当前目录

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import glob
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.cluster import KMeans
from spectro_ablation_experiments import CausalSpectroNet

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        if class_idx is None:
            class_idx = output.argmax().item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "saved_models/ConstantQOnly_seed0.pth"
model = CausalSpectroNet().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
target_layer = model.features[7]
cam_generator = GradCAM(model, target_layer)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def compute_image_difference(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).resize((224, 224))).astype(np.float32) / 255.0
    img2 = np.array(Image.open(img2_path).resize((224, 224))).astype(np.float32) / 255.0
    return np.linalg.norm(img1.flatten() - img2.flatten())

def find_topk_diff_pairs(root="Voice", topk=5):
    output_dirs = [d for d in os.listdir(root) if d.endswith("Output")]
    pd_dirs = [d for d in output_dirs if "_PD_" in d]
    topk_result_all = []
    for pd_dir in pd_dirs:
        pd_path = os.path.join(root, pd_dir)
        dataset_prefix = pd_dir.replace("_PD_Output", "")
        hc_dir = f"{dataset_prefix}_HC_Output"
        hc_path = os.path.join(root, hc_dir)
        if not os.path.exists(hc_path):
            continue
        pd_images = sorted(glob.glob(os.path.join(pd_path, "*_cqt.png")))
        hc_images = sorted(glob.glob(os.path.join(hc_path, "*_cqt.png")))
        local_diffs = [(compute_image_difference(h, p), h, p) for p in pd_images for h in hc_images]
        local_diffs = sorted(local_diffs, reverse=True)[:topk]
        topk_result_all.extend(local_diffs)
    topk_result_all = sorted(topk_result_all, reverse=True)[:topk]
    with open("topk_cqt_pairs_dataset_matched.txt", "w") as f:
        for d, h, p in topk_result_all:
            f.write(f"{h}|{p}\n")
    print("Top-K image differences written.")
find_topk_diff_pairs(topk=7)

topk_image_paths = []
with open("topk_cqt_pairs_dataset_matched.txt", "r") as f:
    for line in f:
        hc_path, pd_path = line.strip().split("|")
        topk_image_paths.extend([hc_path, pd_path])

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model_wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()

def extract_token_heatmap(wav_path, n_clusters=50):
    import torchaudio
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    inputs = processor(wav.squeeze(), return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        out = model_wav2vec(**inputs).last_hidden_state.squeeze(0).numpy()
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(out)
    return km.labels_

def overlay_gradcam_and_token(img_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    cam = cam_generator.generate(input_tensor)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    basename_prefix = os.path.basename(img_path).split("_chunk")[0]
    voice_root = "Voice"
    candidates = []
    for folder in os.listdir(voice_root):
        if folder.endswith("Output") or folder.endswith("Renamed_Files_Output"):
            continue
        matches = glob.glob(os.path.join(voice_root, folder, f"{basename_prefix}*.wav"))
        if matches:
            candidates.extend(matches)
    token_seq = extract_token_heatmap(candidates[0]) if candidates else None
    fig, axs = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 1]})
    axs[0].imshow(img.resize((224, 224)))
    axs[0].imshow(cam, cmap='jet', alpha=0.5)
    axs[0].set_title("GradCAM Overlay")
    axs[0].axis('off')
    if token_seq is not None:
        axs[1].imshow(token_seq[np.newaxis, :], aspect='auto', cmap='tab20')
        axs[1].set_title("Token Heatmap")
        axs[1].set_yticks([])
    output_dir = "figures/gradcam_token_overlay"
    os.makedirs(output_dir, exist_ok=True)
    save_name = os.path.basename(img_path).replace(".png", "_overlay.png")
    plt.savefig(os.path.join(output_dir, save_name), dpi=300)
    plt.close()

for path in topk_image_paths:
    overlay_gradcam_and_token(path)
