"""
NephroLens - Step 8: VM-UNet (Vision Mamba + UNet) with SSM Algorithm
Author: Sanket Kelzarkar
Description: Train VM-UNet Mamba-based model for kidney stone segmentation & classification
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2

# Project Configuration
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
VMUNET_DIR = os.path.join(PROJECT_DIR, "vmunet_training")
RESULTS_DIR = os.path.join(VMUNET_DIR, "results")

os.makedirs(VMUNET_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  MAMBA / SSM CORE COMPONENTS
# ─────────────────────────────────────────────────────────────

class SSMKernel(nn.Module):
    """
    Simplified Structured State Space Model (S4/Mamba-style) kernel.
    State equation:  h_t = A h_{t-1} + B x_t
    Output:          y_t = C h_t + D x_t
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Learnable SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B     = nn.Parameter(torch.randn(d_model, d_state))
        self.C     = nn.Parameter(torch.randn(d_model, d_state))
        self.D     = nn.Parameter(torch.ones(d_model))
        self.dt    = nn.Parameter(torch.ones(d_model) * 0.01)

    def forward(self, x):
        """x: (B, L, d_model)"""
        B, L, d = x.shape
        dt = F.softplus(self.dt)                          # (d,)
        A  = -torch.exp(self.A_log)                       # (d, N)

        # Discretise: zero-order hold
        dA = torch.exp(dt.unsqueeze(-1) * A)              # (d, N)
        dB = dt.unsqueeze(-1) * self.B                    # (d, N)

        # Scan recurrence
        h = torch.zeros(B, d, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            xt = x[:, t, :]                               # (B, d)
            h  = dA.unsqueeze(0) * h + dB.unsqueeze(0) * xt.unsqueeze(-1)
            y  = (h * self.C.unsqueeze(0)).sum(-1)        # (B, d)
            ys.append(y)

        out = torch.stack(ys, dim=1)                      # (B, L, d)
        out = out + self.D.unsqueeze(0).unsqueeze(0) * x
        return out


class MambaBlock(nn.Module):
    """
    One Mamba block:
        LayerNorm → SSM (with selective scan) → Gate → residual
    """
    def __init__(self, d_model, d_state=16, expand=2, dropout=0.1):
        super().__init__()
        d_inner = d_model * expand

        self.norm   = nn.LayerNorm(d_model)
        self.in_proj= nn.Linear(d_model, d_inner * 2)     # x and z
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=3,
                                padding=1, groups=d_inner)
        self.ssm    = SSMKernel(d_inner, d_state)
        self.act    = nn.SiLU()
        self.out_proj= nn.Linear(d_inner, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)                                  # (B, L, d)
        xz = self.in_proj(x)
        xp, z = xz.chunk(2, dim=-1)                       # each (B, L, d_inner)

        # Depthwise conv along sequence
        xp = self.conv1d(xp.transpose(1, 2)).transpose(1, 2)
        xp = self.act(xp)

        xp = self.ssm(xp)
        z  = self.act(z)
        out = self.out_proj(xp * z)
        return self.drop(out) + residual


class VisionMambaEncoder(nn.Module):
    """
    Patch-based Vision Mamba encoder.
    Image → patches → linear embed → stack of Mamba blocks → feature map.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=256, depth=6, d_state=16):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim,
                      kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)                                  # (B, embed, n_patches)
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, embed_dim) * 0.02)

        self.blocks = nn.Sequential(
            *[MambaBlock(embed_dim, d_state) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).transpose(1, 2)           # (B, n_patches, embed)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)                                   # (B, n_patches, embed)
        return x


# ─────────────────────────────────────────────────────────────
# 2.  VM-UNET ARCHITECTURE
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class VMUNet(nn.Module):
    """
    VM-UNet: Vision-Mamba encoder fused with a classic UNet decoder.
    Supports both binary segmentation AND classification.
    """
    def __init__(self, in_channels=3, num_classes=2,
                 img_size=224, embed_dim=256,
                 encoder_depth=4, d_state=16):
        super().__init__()
        # ── CNN down-path (skip connections) ──────────────────
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool  = nn.MaxPool2d(2)

        # ── Mamba bottleneck ──────────────────────────────────
        self.mamba_enc = VisionMambaEncoder(
            img_size=img_size, patch_size=16,
            in_channels=in_channels,
            embed_dim=embed_dim, depth=encoder_depth, d_state=d_state)
        bottleneck_hw = img_size // 16
        self.mamba_proj = nn.Linear(embed_dim, 512)
        self.bottleneck_hw = bottleneck_hw

        # ── CNN up-path (decoder) ─────────────────────────────
        self.up4    = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4   = ConvBlock(512, 256)
        self.up3    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3   = ConvBlock(256, 128)
        self.up2    = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2   = ConvBlock(128, 64)
        self.seg_head = nn.Conv2d(64, 1, 1)                # segmentation output

        # ── Classification head ───────────────────────────────
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, _, H, W = x.shape

        # Encoder (CNN)
        e1 = self.enc1(x)                                  # 64, H, W
        e2 = self.enc2(self.pool(e1))                      # 128, H/2, W/2
        e3 = self.enc3(self.pool(e2))                      # 256, H/4, W/4
        e4 = self.enc4(self.pool(e3))                      # 512, H/8, W/8

        # Mamba bottleneck
        mamba_feat = self.mamba_enc(x)                     # (B, n_patches, embed)
        mamba_feat = self.mamba_proj(mamba_feat)           # (B, n_patches, 512)
        hw = self.bottleneck_hw
        mamba_map  = mamba_feat.transpose(1,2).reshape(B, 512, hw, hw)

        # Fuse CNN e4 with Mamba map
        mamba_map  = F.interpolate(mamba_map, size=e4.shape[2:], mode='bilinear', align_corners=False)
        bottleneck = e4 + mamba_map                        # element-wise fusion

        # Classification output
        cls_out = self.cls_head(bottleneck)

        # Decoder (UNet)
        d4 = self.dec4(torch.cat([self.up4(bottleneck), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        seg_out = torch.sigmoid(self.seg_head(d2))         # binary mask

        return cls_out, seg_out


# ─────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────

class KidneyStoneMambaDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=224):
        self.samples, self.labels = [], []
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5) if split == 'train' else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if split == 'train' else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.2, 0.2) if split == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for lbl, cat in enumerate(['stone', 'normal']):
            d = os.path.join(data_dir, split, cat)
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if f.endswith('.png'):
                    self.samples.append(os.path.join(d, f))
                    self.labels.append(lbl)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        lbl = self.labels[idx]
        # Create pseudo segmentation mask (circular region)
        mask = torch.zeros(1, img.size[1], img.size[0])
        cx, cy, r = img.size[0]//2, img.size[1]//2, min(img.size)//4
        Y, X = torch.meshgrid(torch.arange(img.size[1]), torch.arange(img.size[0]), indexing='ij')
        mask[0] = ((X - cx)**2 + (Y - cy)**2 < r**2).float()
        mask = F.interpolate(mask.unsqueeze(0), size=(224, 224)).squeeze(0)
        return self.transform(img), torch.tensor(lbl, dtype=torch.long), mask


# ─────────────────────────────────────────────────────────────
# 4.  COMBINED LOSS
# ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, cls_weight=1.0, seg_weight=0.5):
        super().__init__()
        self.cls_w  = cls_weight
        self.seg_w  = seg_weight
        self.ce     = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target, eps=1e-6):
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)

    def forward(self, cls_out, seg_out, labels, masks):
        cls_loss = self.ce(cls_out, labels)
        bce_loss = F.binary_cross_entropy(seg_out, masks)
        dice     = self.dice_loss(seg_out, masks)
        seg_loss = bce_loss + dice
        return self.cls_w * cls_loss + self.seg_w * seg_loss, cls_loss, seg_loss


# ─────────────────────────────────────────────────────────────
# 5.  TRAINER
# ─────────────────────────────────────────────────────────────

class VMUNetTrainer:
    def __init__(self, img_size=224, embed_dim=128, encoder_depth=3, d_state=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = VMUNet(
            in_channels=3, num_classes=2,
            img_size=img_size, embed_dim=embed_dim,
            encoder_depth=encoder_depth, d_state=d_state
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {total_params:,}")

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.img_size = img_size

    def prepare_loaders(self, batch_size=16):
        tr = KidneyStoneMambaDataset(PREPROCESSED_DIR, 'train', self.img_size)
        vl = KidneyStoneMambaDataset(PREPROCESSED_DIR, 'val',   self.img_size)
        ts = KidneyStoneMambaDataset(PREPROCESSED_DIR, 'test',  self.img_size)
        kw = dict(num_workers=0, pin_memory=True)
        return (DataLoader(tr, batch_size, shuffle=True,  **kw),
                DataLoader(vl, batch_size, shuffle=False, **kw),
                DataLoader(ts, batch_size, shuffle=False, **kw))

    def _run_epoch(self, loader, criterion, optimizer=None):
        training = optimizer is not None
        self.model.train() if training else self.model.eval()
        tot_loss = correct = total = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc='  Train' if training else '  Val  ')
            for imgs, lbls, masks in pbar:
                imgs  = imgs.to(self.device)
                lbls  = lbls.to(self.device)
                masks = masks.to(self.device)

                cls_out, seg_out = self.model(imgs)
                loss, _, _ = criterion(cls_out, seg_out, lbls, masks)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                tot_loss += loss.item()
                _, pred = cls_out.max(1)
                correct += (pred == lbls).sum().item()
                total   += lbls.size(0)
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 acc=f'{100*correct/total:.1f}%')

        return tot_loss / len(loader), 100 * correct / total

    def train(self, epochs=40, batch_size=16, lr=3e-4):
        print("=" * 60)
        print("NEPHROLENS - VM-UNET (MAMBA SSM) TRAINING")
        print("Powered by Sanket Kelzarkar")
        print("=" * 60)

        train_loader, val_loader, test_loader = self.prepare_loaders(batch_size)
        criterion = CombinedLoss(cls_weight=1.0, seg_weight=0.5)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc  = 0.0
        patience_ctr  = 0
        patience_max  = 10
        best_path     = os.path.join(VMUNET_DIR, 'vmunet_best.pth')

        for ep in range(1, epochs + 1):
            print(f"\nEpoch {ep}/{epochs}  lr={scheduler.get_last_lr()[0]:.6f}")
            tr_loss, tr_acc = self._run_epoch(train_loader, criterion, optimizer)
            vl_loss, vl_acc = self._run_epoch(val_loader,   criterion)
            scheduler.step()

            self.history['train_loss'].append(tr_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_loss'].append(vl_loss)
            self.history['val_acc'].append(vl_acc)

            print(f"  Train → Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%")
            print(f"  Val   → Loss: {vl_loss:.4f}  Acc: {vl_acc:.2f}%")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(self.model.state_dict(), best_path)
                patience_ctr = 0
                print(f"  ✓ New best saved  (Val Acc = {vl_acc:.2f}%)")
            else:
                patience_ctr += 1

            if patience_ctr >= patience_max:
                print(f"\n  Early stopping triggered at epoch {ep}")
                break

        # Load best, evaluate on test
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        ts_loss, ts_acc = self._run_epoch(
            self.prepare_loaders(batch_size)[2], criterion)
        print(f"\n✓ Test Accuracy: {ts_acc:.2f}%")

        self._plot_history()
        self._save_results(ts_acc)

        print("\n" + "=" * 60)
        print("VM-UNET TRAINING COMPLETED!")
        print("Powered by Sanket Kelzarkar")
        print("=" * 60)

        return ts_acc

    def _plot_history(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot with distinct colors and styles
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', lw=2.5, marker='o', 
                markersize=5, color='#1f77b4', linestyle='-')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', lw=2.5, marker='s', 
                markersize=5, color='#ff7f0e', linestyle='--')
        ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax1.set_title('Loss', fontweight='bold', fontsize=12)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        ax1.set_xticks(epochs)

        # Accuracy plot with distinct colors and styles
        ax2.plot(epochs, self.history['train_acc'], label='Train Accuracy', lw=2.5, marker='o', 
                markersize=5, color='#2ca02c', linestyle='-')
        ax2.plot(epochs, self.history['val_acc'], label='Val Accuracy', lw=2.5, marker='s', 
                markersize=5, color='#d62728', linestyle='--')
        ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
        ax2.set_title('Accuracy (%)', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        ax2.set_xticks(epochs)

        plt.suptitle('VM-UNet (Mamba SSM) Training History\nNephroLens – Powered by Sanket Kelzarkar',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'vmunet_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training history plot saved")

    def visualise_segmentation(self, num_samples=6):
        """Visualise predicted segmentation masks on test images."""
        self.model.eval()
        test_ds  = KidneyStoneMambaDataset(PREPROCESSED_DIR, 'test', self.img_size)
        indices  = np.random.choice(len(test_ds), num_samples, replace=False)

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        fig.suptitle('VM-UNet Segmentation Results\nNephroLens – Powered by Sanket Kelzarkar',
                     fontsize=14, fontweight='bold')

        cls_names = ['Stone', 'Normal']
        for row, idx in enumerate(indices):
            img_t, lbl, mask = test_ds[idx]
            with torch.no_grad():
                cls_out, seg_out = self.model(img_t.unsqueeze(0).to(self.device))
            pred_cls  = cls_out.argmax(1).item()
            pred_mask = seg_out[0, 0].cpu().numpy()

            img_np = img_t.permute(1, 2, 0).numpy()
            img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

            axes[row, 0].imshow(img_np); axes[row, 0].set_title('Input CT')
            axes[row, 0].axis('off')
            axes[row, 1].imshow(pred_mask, cmap='hot'); axes[row, 1].set_title('Seg Mask')
            axes[row, 1].axis('off')
            ovl = img_np.copy()
            ovl[:, :, 0] = np.clip(ovl[:, :, 0] + 0.4 * pred_mask, 0, 1)
            axes[row, 2].imshow(ovl)
            axes[row, 2].set_title(f'Overlay – {cls_names[pred_cls]}')
            axes[row, 2].axis('off')

        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, 'vmunet_segmentation_samples.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Segmentation visualisation saved to {out}")

    def _save_results(self, test_acc):
        data = {
            'model': 'VM-UNet (Vision Mamba SSM)',
            'test_accuracy': test_acc,
            'best_val_accuracy': max(self.history['val_acc']),
            'total_epochs': len(self.history['train_loss']),
            'history': self.history
        }
        with open(os.path.join(RESULTS_DIR, 'vmunet_results.json'), 'w') as f:
            json.dump(data, f, indent=4)
        print("✓ Results JSON saved")


# ─────────────────────────────────────────────────────────────
# 6.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    trainer = VMUNetTrainer(img_size=224, embed_dim=128, encoder_depth=3, d_state=8)
    trainer.train(epochs=15, batch_size=8, lr=3e-4)
    trainer.visualise_segmentation(num_samples=6)
    print("\n✓ Ready for next step: Grad-CAM XAI (Step 9)")