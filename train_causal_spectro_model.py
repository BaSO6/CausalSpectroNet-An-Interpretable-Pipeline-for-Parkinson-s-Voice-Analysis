
import os
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- 省略部分说明，为节省篇幅，可补充详细文档 ---

# 灰度方差裁剪
def crop_spectrogram(image, std_ratio=0.1):
    gray = image.convert('L')
    img_np = np.array(gray, dtype=np.float32)
    row_std = np.std(img_np, axis=1)
    col_std = np.std(img_np, axis=0)
    row_thresh = np.max(row_std) * std_ratio
    col_thresh = np.max(col_std) * std_ratio
    rows = np.where(row_std > row_thresh)[0]
    cols = np.where(col_std > col_thresh)[0]
    if rows.size == 0 or cols.size == 0:
        return image
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    return image.crop((left, top, right, bottom))

# 数据集加载类
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        pattern = os.path.join(root_dir, '**', '*Output', '*.png')
        self.image_paths = glob.glob(pattern, recursive=True)
        if len(self.image_paths) == 0:
            raise ValueError("No PNG found.")
        self.labels = [1 if "pd" in p.lower() else 0 for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        spec_type = "constantq" if "constantq" in img_path.lower() else (
                    "scalogram" if "scalogram" in img_path.lower() else "unknown")
        if self.transform:
            image = self.transform(image, spec_type)
        return image, self.labels[idx]

# Global-Local扰动
class GlobalLocalTransform:
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        self.local_transform = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))

    def __call__(self, image, spec_type):
        image = crop_spectrogram(image, std_ratio=0.1)
        image = image.resize(self.output_size, Image.BILINEAR)
        image = transforms.ToTensor()(image)
        image_global = self.global_transform(image, spec_type)
        image_local = self.local_transform(image.clone())
        alpha = random.uniform(0, 1)
        return torch.clamp(alpha * image_global + (1 - alpha) * image_local, 0, 1)

    def global_transform(self, image, spec_type):
        C, H, W = image.shape
        image_fft = torch.fft.fft2(image)
        cx, cy = W // 2, H // 2
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_x = grid_x.to(image.device)
        grid_y = grid_y.to(image.device)
        distance = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
        r = random.uniform(0.1, 0.3) * max(W, H)
        if spec_type == "scalogram":
            r *= 1.5
        mask = (distance < r).float().unsqueeze(0).repeat(C, 1, 1)
        noise = torch.randn_like(image_fft) * 0.1
        image_fft_aug = image_fft * mask + noise
        return torch.fft.ifft2(image_fft_aug).real

# CausalSpectroNet结构
class CausalSpectroNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CausalSpectroNet, self).__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.attention = nn.Conv2d(512, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        att_map = torch.sigmoid(self.attention(features))
        pooled = self.avgpool(features * att_map).view(x.size(0), -1)
        return self.fc(pooled), att_map

# 损失函数
def dice_loss(att1, att2, eps=1e-6):
    att1_bin = (att1 > 0.5).float()
    att2_bin = (att2 > 0.5).float()
    intersection = (att1_bin * att2_bin).sum(dim=[1, 2, 3])
    union = att1_bin.sum(dim=[1, 2, 3]) + att2_bin.sum(dim=[1, 2, 3])
    return 1 - (2 * intersection + eps) / (union + eps)

def kl_div_loss(p_logits, q_logits):
    p = torch.softmax(p_logits, dim=1)
    log_q = torch.log_softmax(q_logits, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(log_q, p)

# 训练与验证
def train_epoch(model, dataloader, optimizer, device, weights, lambda_att=0.1, lambda_proto=0.1):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits1, att1 = model(images)
        logits2, att2 = model(images.clone())
        loss = criterion(logits1, labels) + criterion(logits2, labels) +                lambda_att * dice_loss(att1, att2) +                lambda_proto * (kl_div_loss(logits1, logits2) + kl_div_loss(logits2, logits1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, device, weights):
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            total_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Val Acc: {correct / total * 100:.2f}%")
    return total_loss / len(dataloader)

# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpectrogramDataset(root_dir="./Voice", transform=GlobalLocalTransform())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16)

    counts = [dataset.labels.count(i) for i in range(2)]
    beta = 0.9999
    weights = torch.tensor([(1 - beta) / (1 - beta ** c) for c in counts])
    weights = weights / weights.sum() * 2

    model = CausalSpectroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        print(f"Epoch {epoch + 1}")
        train_loss = train_epoch(model, train_loader, optimizer, device, weights)
        print(f"Train Loss: {train_loss:.4f}")
        validate(model, val_loader, device, weights)

    torch.save(model.state_dict(), "causal_spectro_model.pth")
    print("✅ Model saved to causal_spectro_model.pth")

if __name__ == "__main__":
    main()
