# spectro_ablation_experiments_single.py
# 单模型模式运行版，增加输出文件保存 (model和log)

import os, random, glob, argparse, json, time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 随机性
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据集
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, spec_mode="multi"):
        pattern = os.path.join(root_dir, "**", "*Output", "*.png")
        all_paths = glob.glob(pattern, recursive=True)
        def keep(p):
            low = p.lower()
            if spec_mode == "multi": return True
            if spec_mode == "constantq": return "constantq" in low or "cqt" in low
            if spec_mode == "scalogram": return "scalogram" in low or "cwt" in low
            return False
        self.image_paths = [p for p in all_paths if keep(p)]
        self.labels = [1 if ("/pd" in p.lower() or "_pd" in p.lower()) else 0 for p in self.image_paths]
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        spec_type = "constantq" if "constantq" in path.lower() else ("scalogram" if "scalogram" in path.lower() else "unknown")
        return self.transform(img, spec_type), self.labels[idx]

# 全局+局部增强
class GlobalLocalTransform:
    def __init__(self, size=(224,224), enable_aug=True):
        self.output_size = size
        self.enable_aug = enable_aug
        self.local_aug = T.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3))

    def __call__(self, image: Image.Image, spec_type: str):
        image = T.ToTensor()(image.resize(self.output_size))
        if not self.enable_aug: return image
        img_g = self._global_freq_aug(image.clone(), spec_type)
        img_l = self.local_aug(image.clone())
        alpha = random.random()
        return torch.clamp(alpha*img_g + (1-alpha)*img_l, 0, 1)

    def _global_freq_aug(self, x, spec_type):
        C,H,W = x.shape
        fft = torch.fft.fft2(x); cx,cy = W//2, H//2
        yy,xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        dist = ((yy-cy)**2 + (xx-cx)**2).sqrt()
        r = random.uniform(0.1,0.2)*max(H,W)
        mask = (dist<r).float().unsqueeze(0).repeat(C,1,1)
        noise = torch.randn_like(fft)*0.1
        return torch.fft.ifft2(fft*mask + noise).real

# 模型
class CausalSpectroNet(nn.Module):
    def __init__(self, use_causal=True):
        super().__init__()
        base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)
        self.use_causal = use_causal
        if use_causal:
            self.att = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        feat = self.features(x)
        if self.use_causal:
            att = torch.sigmoid(self.att(feat))
            feat = feat * att
        out = self.pool(feat).flatten(1)
        return self.fc(out)

# 训练 / 温泉

def train_one_epoch(model, loader, opt, device, cls_w):
    model.train(); ce = nn.CrossEntropyLoss(weight=cls_w.to(device)); total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); logits = model(x)
        loss = ce(logits, y); loss.backward(); opt.step(); total += loss.item()
    return total / len(loader)

def evaluate(model, loader, device, cls_w):
    model.eval(); correct, total, loss_sum = 0, 0, 0
    ce = nn.CrossEntropyLoss(weight=cls_w.to(device))
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += ce(out,y).item()
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)
    return loss_sum/len(loader), correct/total

# 单个 config 运行

def run_single_cfg(cfg_name, data_root, epochs, seed):
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cfg_map = {
        "Baseline-Full":     (True, "multi",  True),
        "-GlobalLocal":      (False,"multi",  True),
        "ConstantQ-Only":    (True, "constantq", True),
        "Scalogram-Only":    (True, "scalogram", True),
        "-CausalAtt":        (True, "multi",  False),
    }
    aug_flag, spec_mode, use_causal = cfg_map[cfg_name]
    transform = GlobalLocalTransform(enable_aug=aug_flag)

    dataset = SpectrogramDataset(data_root, transform, spec_mode=spec_mode)
    print(f"[Info] Loaded {len(dataset)} samples for config: {cfg_name}")

    n = len(dataset); n_train = int(n*0.8); n_val = int(n*0.1)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n-n_train-n_val])

    cls_weights = torch.tensor([
        1 - sum(y==1 for _,y in train_set)/len(train_set),
        1 - sum(y==0 for _,y in train_set)/len(train_set)
    ])

    dl_train = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    dl_val = torch.utils.data.DataLoader(val_set, batch_size=16)
    dl_test = torch.utils.data.DataLoader(test_set, batch_size=16)

    model = CausalSpectroNet(use_causal=use_causal).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    best_state, best_val_acc = None, 0
    log_lines = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, dl_train, opt, device, cls_weights)
        val_loss, val_acc = evaluate(model, dl_val, device, cls_weights)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={val_acc*100:.2f}%")
        log_lines.append(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, dl_test, device, cls_weights)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    log_lines.append(f"Test accuracy: {test_acc*100:.2f}%")

    # 保存模型 & 日志
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    model_path = f"saved_models/{cfg_name.replace('-', '').replace(' ', '_')}_seed{seed}.pth"
    with open(f"logs/{cfg_name.replace('-', '').replace(' ', '_')}_seed{seed}.log", "w") as f:
        f.write("\n".join(log_lines))
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_name", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_single_cfg(args.cfg_name, args.data_root, args.epochs, args.seed)