# token_model_training.py
# 用 CNN/Transformer 对 audio token 序列进行分类

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import numpy as np

# ---------------------------
# Dataset: 加载 token 序列
# ---------------------------
class TokenSequenceDataset(Dataset):
    def __init__(self, csv_path, max_len=128):
        df = pd.read_csv(csv_path)
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
        self.samples = []
        for _, row in df.iterrows():
            tokens = row["tokens"][:max_len]
            if len(tokens) < max_len:
                tokens += [0] * (max_len - len(tokens))  # PAD
            label = int(row["label"])
            self.samples.append((tokens, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ---------------------------
# Token 模型结构 (CNN + Embed)
# ---------------------------
class TokenCNNClassifier(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=64, max_len=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embed(x)             # (B, T, E)
        x = x.permute(0, 2, 1)        # (B, E, T)
        x = self.conv(x)              # (B, 256, 1)
        x = x.squeeze(-1)             # (B, 256)
        return self.fc(x)

# ---------------------------
# 训练 / 验证函数
# ---------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

# ---------------------------
# 主程序：训练 + 验证
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "audio_vq_multimodal_with_llm.csv"
    batch_size = 32
    max_len = 128
    
    full_dataset = TokenSequenceDataset(data_path, max_len=max_len)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = TokenCNNClassifier(vocab_size=30, embed_dim=64, max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | Val Loss={val_loss:.4f}, Acc={val_acc:.2%}")

    torch.save(model.state_dict(), "token_cnn_model.pth")
    print("✅ Token-based CNN 模型训练完成并已保存。")

if __name__ == "__main__":
    main()