
import os, random, glob, argparse, json, time
from dataclasses import dataclass, asdict
from typing import List, Dict

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from PIL import Image
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

########################################
# Reproducibility
########################################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################################
# Dataset with optional JSON metadata
########################################

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transform, spec_mode: str = "multi"):
        self.root_dir, self.transform, self.spec_mode = root_dir, transform, spec_mode
        pattern = os.path.join(root_dir, "**", "*Output", "*.png")
        all_paths = glob.glob(pattern, recursive=True)

        def keep(p):
            low = p.lower()
            if spec_mode == "multi":
                return True
            if spec_mode == "constantq":
                return "constantq" in low or "cqt" in low
            if spec_mode == "scalogram":
                return "scalogram" in low or "cwt" in low
            raise ValueError("Unknown spec_mode", spec_mode)

        self.image_paths = [p for p in all_paths if keep(p)]
        self.labels = [1 if ("/pd" in p.lower() or "_pd" in p.lower()) else 0 for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)  # ✅ 新增：加速图像处理（避免大图）

        json_path = img_path.replace(".png", "_spec.json")
        metadata = {}
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                metadata = json.load(f)

        spec_type = (
            "constantq" if "constantq" in img_path.lower() else
            "scalogram" if "scalogram" in img_path.lower() else
            "unknown"
        )
        image = self.transform(image, spec_type)
        return image, self.labels[idx], metadata

########################################
# Global-Local Transform (optional)
########################################

class GlobalLocalTransform:
    def __init__(self, output_size=(224,224), enable_aug: bool = True):
        self.output_size = output_size
        self.enable_aug  = enable_aug
        self.local_aug   = T.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3))

    def __call__(self, image: Image.Image, spec_type: str):
        image = image.resize(self.output_size, Image.BILINEAR)
        image = T.ToTensor()(image)
        if not self.enable_aug:
            return image
        img_g = self._global_freq_aug(image.clone(), spec_type)
        img_l = self.local_aug(image.clone())
        alpha = random.random()
        return torch.clamp(alpha*img_g + (1-alpha)*img_l, 0, 1)

    def _global_freq_aug(self, x: torch.Tensor, spec_type: str):
        C,H,W = x.shape
        fft = torch.fft.fft2(x)
        cx,cy = W//2, H//2
        yy,xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        dist  = ((yy-cy).float()**2 + (xx-cx).float()**2).sqrt()
        r_fac = 0.2 if spec_type=="scalogram" else 0.15
        r     = random.uniform(r_fac*0.5, r_fac*1.5)*max(H,W)
        mask  = (dist<r).float().unsqueeze(0).repeat(C,1,1)
        noise = torch.randn_like(fft)*0.1
        aug   = torch.fft.ifft2(fft*mask + noise).real
        return aug

########################################
# Model with optional causal attention
########################################

class CausalSpectroNet(nn.Module):
    def __init__(self, num_classes=2, use_causal: bool = True):
        super().__init__()
        self.use_causal = use_causal
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        if use_causal:
            self.att = nn.Conv2d(512,1,1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.features(x)
        if self.use_causal:
            att  = torch.sigmoid(self.att(feat))
            feat = feat * att
        else:
            att  = torch.zeros_like(feat[:, :1])
        out = self.pool(feat).flatten(1)
        return self.fc(out), att

########################################
# Train & Evaluate
########################################

def train_one_epoch(model, loader, opt, device, cls_w):
    model.train()
    ce = nn.CrossEntropyLoss(weight=cls_w.to(device))
    tot = 0
    for batch in loader:
        if len(batch) == 3:
            imgs, lbls, _ = batch
        else:
            imgs, lbls = batch
        imgs,lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        log1,att1 = model(imgs)
        log2,att2 = model(imgs)
        loss = ce(log1,lbls)+ce(log2,lbls)
        loss.backward(); opt.step(); tot+=loss.item()
    return tot/len(loader)

def evaluate(model, loader, device, cls_w):
    model.eval()
    ce = nn.CrossEntropyLoss(weight=cls_w.to(device))
    tot,correct,n = 0,0,0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, lbls, _ = batch
            else:
                imgs, lbls = batch
            imgs,lbls = imgs.to(device), lbls.to(device)
            log,_ = model(imgs)
            tot+=ce(log,lbls).item()
            preds=log.argmax(1)
            correct+=(preds==lbls).sum().item()
            n+=lbls.size(0)
    return tot/len(loader), correct/n

########################################
# Config
########################################

@dataclass
class ExpCfg:
    name: str
    use_global_local: bool = True
    spec_mode: str = "multi"
    use_causal: bool = True

DEFAULT_CFGS = [
    ExpCfg("Baseline‑Full"),
    ExpCfg("‑GlobalLocal", use_global_local=False),
    ExpCfg("ConstantQ‑Only", spec_mode="constantq"),
    ExpCfg("Scalogram‑Only", spec_mode="scalogram"),
    ExpCfg("‑CausalAtt", use_causal=False),
]

########################################
# Main loop with statistics
########################################

def run_experiments(data_root: str, epochs: int, seeds: List[int], cfgs: List[ExpCfg]):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    results = {cfg.name: [] for cfg in cfgs}

    for seed in seeds:
        set_seed(seed)
        for cfg in cfgs:
            print(f"\n===== Seed {seed} | Config {cfg.name} =====")
            trans = GlobalLocalTransform(enable_aug=cfg.use_global_local)
            ds_full = SpectrogramDataset(data_root, trans, spec_mode=cfg.spec_mode)
            print(f"[Info] Loaded {len(ds_full)} samples for {cfg.name}")
            if len(ds_full) < 10:
                print(f"[Warning] Skipping config {cfg.name} due to insufficient samples: {len(ds_full)}")
                continue
            n=len(ds_full); n_train=int(0.8*n); n_val=int(0.1*n)
            train,val,test = torch.utils.data.random_split(ds_full,[n_train,n_val,n-n_train-n_val])
            w = torch.tensor([ (1-len([l for _,l,_ in train if l==1])/len(train)),
                               (1-len([l for _,l,_ in train if l==0])/len(train)) ])
            dl_train = torch.utils.data.DataLoader(train,16,True)
            dl_val   = torch.utils.data.DataLoader(val,16,False)
            dl_test  = torch.utils.data.DataLoader(test,16,False)

            model = CausalSpectroNet(use_causal=cfg.use_causal).to(device)
            opt   = optim.Adam(model.parameters(),1e-4)
            best_acc=0
            for ep in range(epochs):
                tr_loss = train_one_epoch(model, dl_train, opt, device, w)
                _, val_acc = evaluate(model, dl_val, device, w)
                if val_acc>best_acc:
                    best_acc=val_acc; best_state=model.state_dict().copy()
                print(f"Epoch {ep+1}/{epochs}  trainLoss={tr_loss:.3f}  valAcc={val_acc*100:.2f}%")
            model.load_state_dict(best_state)
            _, test_acc = evaluate(model, dl_test, device, w)
            print(f"Test accuracy: {test_acc*100:.2f}%")
            results[cfg.name].append(test_acc)

            # 实时保存中间结果
            df_interim = pd.DataFrame({k: pd.Series(v) for k,v in results.items()})
            df_interim.to_csv("ablation_results_interim.csv")

    df = pd.DataFrame({k: pd.Series(v) for k,v in results.items()})
    means, stds = df.mean()*100, df.std()*100
    baseline = results[cfgs[0].name]
    pvals = {k: stats.ttest_ind(baseline, v, equal_var=False).pvalue if k!=cfgs[0].name else np.nan for k,v in results.items()}
    summary = pd.DataFrame({"mean%":means, "std%":stds, "p_vs_base":pvals})
    print("\n===== Summary (mean±std,  p vs baseline) =====")
    print(summary.round(2))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary.to_csv(f"ablation_results_{timestamp}.csv")
    summary["mean%"].plot.bar(yerr=summary["std%"], capsize=4)
    plt.ylabel("Accuracy (%)")
    plt.title("Ablation Study")
    plt.tight_layout()
    plt.savefig(f"ablation_bar_{timestamp}.png", dpi=300)

########################################
# CLI
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0,1,2])
    parser.add_argument("--cfg_json", help="Optional path to custom cfg list in JSON")
    args = parser.parse_args()

    cfgs = DEFAULT_CFGS
    if args.cfg_json:
        with open(args.cfg_json) as f:
            cfgs = [ExpCfg(**d) for d in json.load(f)]
    run_experiments(args.data_root, args.epochs, args.seeds, cfgs)