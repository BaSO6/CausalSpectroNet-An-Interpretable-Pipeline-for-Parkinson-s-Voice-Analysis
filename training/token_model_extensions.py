# token_model_extensions.py
# 包含三个扩展模块：Transformer 模型 / 混淆矩阵评估 / Token+Spectrogram 融合模型

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --------- 1. Transformer Token 模型 ----------
class TokenTransformerClassifier(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=64, max_len=128, num_classes=2, nhead=4, nlayers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embed(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# --------- 2. 评估函数：混淆矩阵 + 分类报告 ----------
def evaluate_with_confusion_matrix(model, dataloader, device, class_names=["HC", "PD"]):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    print("\n[Classification Report]:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("[Confusion Matrix]:")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Token-based Model Confusion Matrix")
    plt.tight_layout()
    plt.show()

# --------- 3. Token + Spectrogram 多模态模型 ----------
class MultiModalClassifier(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=64, token_len=128, spectro_input_shape=(1, 224, 224), num_classes=2):
        super().__init__()
        self.token_branch = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim, padding_idx=0),
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.spectro_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128 + 32, num_classes)

    def forward(self, token_seq, spectrogram):
        x_tok = self.token_branch[0](token_seq)  # embedding
        x_tok = self.token_branch[1:](x_tok.permute(0, 2, 1)).squeeze(-1)  # (B, 128)

        x_spec = self.spectro_branch(spectrogram).squeeze(-1).squeeze(-1)  # (B, 32)
        x = torch.cat([x_tok, x_spec], dim=1)
        return self.fc(x)
import torch
import torch.nn as nn

class TokenCNN(nn.Module):
    def __init__(self, num_tokens=30, emb_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, emb_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):  # x: (B, T)
        x = self.embed(x).transpose(1, 2)  # (B, emb_dim, T)
        x = self.conv(x).squeeze(-1)       # (B, 128)
        out = self.fc(x)                   # (B, 2)
        return out
