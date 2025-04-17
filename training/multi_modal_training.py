# multi_modal_training.py
# 使用 MultiModalClassifier 模型：融合 token + spectrogram 图像训练

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import os
from token_model_extensions import MultiModalClassifier, evaluate_with_confusion_matrix

# ---------------------------
# 融合数据集：Token + Spectrogram
# ---------------------------
class TokenSpectroDataset(Dataset):
    def __init__(self, csv_path, image_root, max_len=128, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.df["tokens"] = self.df["tokens"].apply(ast.literal_eval)
        self.image_root = image_root
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row["tokens"][:self.max_len]
        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        # 加载对应的 spectrogram 图像
        filename = row["filename"].replace(".wav", "_constantq.png")  # 默认 constantq 图
        image_path = os.path.join(self.image_root, filename)
        image = Image.open(image_path).convert("L")  # 单通道
        image_tensor = self.transform(image).unsqueeze(0)  # (1, H, W)

        label = int(row["label"])
        return token_tensor, image_tensor, torch.tensor(label, dtype=torch.long)

# ---------------------------
# 训练与验证函数
# ---------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for token, image, label in loader:
        token, image, label = token.to(device), image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(token, image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(dim=1) == label).sum().item()
        total += label.size(0)
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for token, image, label in loader:
            token, image, label = token.to(device), image.to(device), label.to(device)
            output = model(token, image)
            loss = criterion(output, label)
            total_loss += loss.item()
            correct += (output.argmax(dim=1) == label).sum().item()
            total += label.size(0)
    return total_loss / len(loader), correct / total

# ---------------------------
# 主程序：训练 MultiModalClassifier
# ---------------------------
def main():
    csv_path = "audio_vq_multimodal_with_llm.csv"
    image_root = "Voice_output_dir"  # 修改为实际图像根目录路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TokenSpectroDataset(csv_path, image_root=image_root)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    model = MultiModalClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | Val Loss={val_loss:.4f}, Acc={val_acc:.2%}")

    torch.save(model.state_dict(), "multi_modal_model.pth")
    print("\n✅ 多模态模型训练完成，已保存为 multi_modal_model.pth")

    # 可选：可视化结果
    evaluate_with_confusion_matrix(model, val_loader, device)

if __name__ == "__main__":
    main()
