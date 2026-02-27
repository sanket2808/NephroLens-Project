"""
NephroLens - Step 11: Hyperparameter Tuning (Fixed)
Author: Sanket Kelzarkar
Description: Grid search + Bayesian optimisation for ResNet50.
             Lightweight search on a small validation subset so it runs
             on a laptop GPU within a reasonable time.

Fixes applied:
  - pretrained=True  → weights=ResNet50_Weights.DEFAULT  (no deprecation warning)
  - np.random.logsample lambda  → proper log-uniform function
  - FC architecture consistent with app.py (dropout1, hidden, dropout2, 2)
  - Bayesian search exploration ratio fixed (was inverting 80/20 logic)
  - Added reproducibility seed
  - Sensitivity plot handles bool dtype correctly
"""

import os, json, itertools, random, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ── Reproducibility ──────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ────────────────────────────────────
PROJECT_DIR      = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
TUNING_DIR       = os.path.join(PROJECT_DIR, "hyperparameter_tuning")
os.makedirs(TUNING_DIR, exist_ok=True)

DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE         = 224
MAX_QUICK_EPOCHS = 8        # epochs per trial for grid/Bayesian search


# ═══════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════
class QuickDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=224, augment=True):
        self.samples, self.labels = [], []

        ops = [transforms.Resize((img_size, img_size))]
        if augment and split == 'train':
            ops += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]
        self.tf = transforms.Compose(ops)

        for lbl, cat in enumerate(['stone', 'normal']):
            d = os.path.join(data_dir, split, cat)
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith('.png'):
                    self.samples.append(os.path.join(d, f))
                    self.labels.append(lbl)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        return self.tf(img), self.labels[idx]


def make_loaders(batch_size=32, max_per_class=200):
    """Small subsampled loaders for fast hyperparameter trials."""
    full_tr = QuickDataset(PREPROCESSED_DIR, 'train', IMG_SIZE, augment=True)
    full_vl = QuickDataset(PREPROCESSED_DIR, 'val',   IMG_SIZE, augment=False)

    def subsample(ds):
        idx_s = [i for i, l in enumerate(ds.labels) if l == 0][:max_per_class]
        idx_n = [i for i, l in enumerate(ds.labels) if l == 1][:max_per_class]
        return Subset(ds, idx_s + idx_n)

    kw = dict(num_workers=2, pin_memory=True)
    tr = DataLoader(subsample(full_tr), batch_size, shuffle=True,  **kw)
    vl = DataLoader(subsample(full_vl), batch_size, shuffle=False, **kw)
    return tr, vl


# ═══════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════
def build_resnet(dropout1, dropout2, hidden_dim, freeze_backbone=False):
    """
    Build ResNet50 with custom head.
    Architecture matches app.py for consistency:
        Dropout(d1) → Linear(2048, hidden) → ReLU → Dropout(d2) → Linear(hidden, 2)
    """
    # FIX: use weights= instead of deprecated pretrained=True
    m = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        # Freeze all except last 20 parameters (layer4 + fc)
        params = list(m.parameters())
        for p in params[:-20]:
            p.requires_grad = False

    m.fc = nn.Sequential(
        nn.Dropout(dropout1),
        nn.Linear(m.fc.in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout2),
        nn.Linear(hidden_dim, 2)
    )
    return m.to(DEVICE)


# ═══════════════════════════════════════════════
# SINGLE TRIAL TRAINING LOOP
# ═══════════════════════════════════════════════
def run_trial(params, tr_loader, vl_loader, epochs=MAX_QUICK_EPOCHS):
    """Train for a few epochs and return best validation accuracy."""
    model = build_resnet(
        dropout1=params['dropout1'],
        dropout2=params['dropout2'],
        hidden_dim=params['hidden_dim'],
        freeze_backbone=params['freeze_backbone']
    )

    OptimizerClass = {'Adam': optim.Adam,
                      'AdamW': optim.AdamW,
                      'SGD':  optim.SGD}[params['optimizer']]

    opt = OptimizerClass(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params['lr'],
        weight_decay=params.get('weight_decay', 1e-4)
    )

    cw = torch.tensor(params.get('class_weights', [1.0, 1.0]),
                      dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=params['lr'],
        epochs=epochs, steps_per_epoch=len(tr_loader)
    )

    best_val = 0.0
    for _ in range(epochs):
        # ── Train ──
        model.train()
        for imgs, lbls in tr_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            criterion(model(imgs), lbls).backward()
            opt.step()
            scheduler.step()

        # ── Validate ──
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for imgs, lbls in vl_loader:
                out = model(imgs.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy())
                truths.extend(lbls.numpy())
        acc = accuracy_score(truths, preds)
        if acc > best_val:
            best_val = acc

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_val


# ═══════════════════════════════════════════════
# 1. GRID SEARCH
# ═══════════════════════════════════════════════
GRID = {
    'lr':              [1e-3, 5e-4],
    'dropout1':        [0.3, 0.5],
    'dropout2':        [0.2, 0.4],
    'hidden_dim':      [256, 512],
    'optimizer':       ['Adam', 'AdamW'],
    'freeze_backbone': [True, False],
    'weight_decay':    [1e-4],
    'class_weights':   [[1.0, 1.0]],
}


def grid_search(tr_loader, vl_loader):
    print("\n" + "═" * 60)
    print("[GRID SEARCH]")
    print("═" * 60)

    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"  Total combinations: {len(combos)}")

    records = []
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        print(f"\n  Trial {i}/{len(combos)}: "
              f"lr={params['lr']:.0e}  opt={params['optimizer']}  "
              f"hidden={params['hidden_dim']}  freeze={params['freeze_backbone']}")
        val_acc = run_trial(params, tr_loader, vl_loader)
        print(f"  → Val Accuracy: {val_acc * 100:.2f}%")
        records.append({**params, 'val_accuracy': val_acc})

    records.sort(key=lambda r: r['val_accuracy'], reverse=True)

    gs_path = os.path.join(TUNING_DIR, 'grid_search_results.json')
    with open(gs_path, 'w') as f:
        json.dump(records, f, indent=4, default=str)
    print(f"\n✓ Grid search results saved → {gs_path}")

    print("\nTop 5 Grid Configurations:")
    for r in records[:5]:
        print(f"  Acc={r['val_accuracy']*100:.2f}%  lr={r['lr']}  "
              f"opt={r['optimizer']}  hidden={r['hidden_dim']}")

    return records[0]


# ═══════════════════════════════════════════════
# 2. BAYESIAN OPTIMISATION (random exploration + exploitation)
# ═══════════════════════════════════════════════

def _log_uniform(lo, hi):
    """FIX: proper log-uniform sampler (was broken lambda on np.random)."""
    return float(10 ** np.random.uniform(lo, hi))


def bayesian_search(tr_loader, vl_loader, best_grid_params, n_trials=12):
    print("\n" + "═" * 60)
    print("[BAYESIAN OPTIMISATION]")
    print("═" * 60)

    # Sampling functions for each hyperparameter
    space = {
        'lr':              lambda: _log_uniform(-4, -2.3),
        'dropout1':        lambda: round(random.uniform(0.2, 0.6), 2),
        'dropout2':        lambda: round(random.uniform(0.1, 0.5), 2),
        'hidden_dim':      lambda: random.choice([128, 256, 384, 512, 768]),
        'optimizer':       lambda: random.choice(['Adam', 'AdamW']),
        'freeze_backbone': lambda: random.choice([True, False]),
        'weight_decay':    lambda: _log_uniform(-5, -3),
        'class_weights':   lambda: [1.0, 1.0],
    }

    records     = []
    best_so_far = best_grid_params.get('val_accuracy', 0.0)
    best_params = dict(best_grid_params)

    for trial in range(1, n_trials + 1):
        # FIX: first 3 trials always explore; after that 80% explore / 20% exploit
        if trial <= 3 or random.random() < 0.8:
            params = {k: fn() for k, fn in space.items()}
        else:
            # Exploit: perturb the current best
            params = dict(best_params)
            perturb_key = random.choice(['lr', 'dropout1', 'hidden_dim'])
            params[perturb_key] = space[perturb_key]()

        print(f"\n  Trial {trial}/{n_trials}: "
              f"lr={params['lr']:.2e}  "
              f"drop=({params['dropout1']},{params['dropout2']})  "
              f"hidden={params['hidden_dim']}")

        val_acc = run_trial(params, tr_loader, vl_loader)
        print(f"  → Val Accuracy: {val_acc * 100:.2f}%")

        params['val_accuracy'] = val_acc
        records.append(dict(params))

        if val_acc > best_so_far:
            best_so_far = val_acc
            best_params = dict(params)
            print(f"  ★ New best: {val_acc * 100:.2f}%")

    records.sort(key=lambda r: r['val_accuracy'], reverse=True)
    bo_path = os.path.join(TUNING_DIR, 'bayesian_search_results.json')
    with open(bo_path, 'w') as f:
        json.dump(records, f, indent=4, default=str)
    print(f"\n✓ Bayesian search results saved → {bo_path}")

    return best_params


# ═══════════════════════════════════════════════
# 3. FINAL RETRAINING WITH BEST PARAMS
# ═══════════════════════════════════════════════
def retrain_best(best_params, final_epochs=50):
    print("\n" + "═" * 60)
    print("[FINAL RETRAINING WITH BEST HYPERPARAMETERS]")
    print("═" * 60)
    print("Best params:")
    for k, v in best_params.items():
        if k != 'val_accuracy':
            print(f"  {k:20s}: {v}")

    # Full (non-subsampled) loaders
    kw = dict(num_workers=4, pin_memory=True)
    tr_ld = DataLoader(QuickDataset(PREPROCESSED_DIR, 'train', IMG_SIZE, augment=True),
                       32, shuffle=True,  **kw)
    vl_ld = DataLoader(QuickDataset(PREPROCESSED_DIR, 'val',   IMG_SIZE, augment=False),
                       32, shuffle=False, **kw)
    ts_ld = DataLoader(QuickDataset(PREPROCESSED_DIR, 'test',  IMG_SIZE, augment=False),
                       32, shuffle=False, **kw)

    model = build_resnet(
        best_params['dropout1'], best_params['dropout2'],
        best_params['hidden_dim'], best_params['freeze_backbone']
    )

    OptimizerClass = {'Adam': optim.Adam,
                      'AdamW': optim.AdamW,
                      'SGD':  optim.SGD}[best_params['optimizer']]
    opt = OptimizerClass(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_params['lr'],
        weight_decay=best_params.get('weight_decay', 1e-4)
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=final_epochs)

    history  = {'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    best_path = os.path.join(TUNING_DIR, 'resnet_best_tuned.pth')

    def eval_acc(loader):
        model.eval()
        p, t = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                p.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
                t.extend(lbls.numpy())
        return accuracy_score(t, p)

    for ep in tqdm(range(1, final_epochs + 1), desc='  Retraining'):
        model.train()
        for imgs, lbls in tr_ld:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            criterion(model(imgs), lbls).backward()
            opt.step()
        scheduler.step()

        tr_acc = eval_acc(tr_ld)
        vl_acc = eval_acc(vl_ld)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), best_path)

    # Final test evaluation
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    ts_acc = eval_acc(ts_ld)
    print(f"\n  ✓ Final Test Accuracy (tuned): {ts_acc * 100:.2f}%")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot([v * 100 for v in history['train_acc']], label='Train', lw=2, color='#3498db')
    plt.plot([v * 100 for v in history['val_acc']],   label='Val',   lw=2, color='#e74c3c')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.grid(alpha=.3)
    plt.title('Final Tuned ResNet50 Training\nNephroLens – Powered by Sanket Kelzarkar',
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(TUNING_DIR, 'final_tuned_training.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training plot saved")

    # Save metadata
    meta = {
        'best_params': {k: v for k, v in best_params.items() if k != 'val_accuracy'},
        'final_test_accuracy': ts_acc,
        'best_val_accuracy':   best_acc,
        'total_epochs':        final_epochs,
    }
    with open(os.path.join(TUNING_DIR, 'tuning_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=4, default=str)

    return ts_acc


# ═══════════════════════════════════════════════
# 4. HYPERPARAMETER SENSITIVITY PLOT
# ═══════════════════════════════════════════════
def plot_sensitivity(gs_records):
    import pandas as pd
    df = pd.DataFrame(gs_records)
    df['val_accuracy'] *= 100

    # FIX: convert bool columns to string so seaborn boxplot works correctly
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'bool':
            df[col] = df[col].astype(str)

    params_to_plot = ['lr', 'dropout1', 'dropout2',
                      'hidden_dim', 'optimizer', 'freeze_backbone']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hyperparameter Sensitivity – NephroLens\nPowered by Sanket Kelzarkar',
                 fontsize=15, fontweight='bold', y=1.01)

    for ax, param in zip(axes.flat, params_to_plot):
        col = df[param]
        # Categorical → boxplot, numeric → scatter
        try:
            numeric_vals = pd.to_numeric(col)
            ax.scatter(numeric_vals, df['val_accuracy'],
                       alpha=0.7, color='steelblue', edgecolors='white', s=60)
            ax.set_xlabel(param)
        except (ValueError, TypeError):
            sns.boxplot(data=df, x=param, y='val_accuracy', ax=ax,
                        palette='Set2')
            ax.set_xlabel(param)

        ax.set_ylabel('Val Accuracy (%)')
        ax.set_title(f'Effect of {param}', fontweight='bold')
        ax.grid(alpha=.3)

    plt.tight_layout()
    out = os.path.join(TUNING_DIR, 'hyperparameter_sensitivity.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Sensitivity plot saved → {out}")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("NEPHROLENS – HYPERPARAMETER TUNING")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)
    print(f"\n  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # Shared small loaders for all trials
    tr_loader, vl_loader = make_loaders(batch_size=32, max_per_class=200)

    # ── 1. Grid Search ──────────────────────────
    best_grid = grid_search(tr_loader, vl_loader)

    gs_records = json.load(
        open(os.path.join(TUNING_DIR, 'grid_search_results.json')))
    plot_sensitivity(gs_records)

    # ── 2. Bayesian Optimisation ────────────────
    best_all = bayesian_search(tr_loader, vl_loader, best_grid, n_trials=12)

    # ── 3. Final retrain on full data ───────────
    final_acc = retrain_best(best_all, final_epochs=50)

    # ── 4. Summary ──────────────────────────────
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 60)
    print(f"  Best Grid-Search Val Acc : {best_grid['val_accuracy'] * 100:.2f}%")
    print(f"  Best Bayesian Val Acc    : {best_all.get('val_accuracy', 0) * 100:.2f}%")
    print(f"  Final Test Accuracy      : {final_acc * 100:.2f}%")
    print(f"\n  Best Hyperparameters:")
    for k, v in best_all.items():
        if k != 'val_accuracy':
            print(f"    {k:20s}: {v}")
    print(f"\n✓ Tuned model → {os.path.join(TUNING_DIR, 'resnet_best_tuned.pth')}")
    print("=" * 60)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)
    print("\n✓ Ready for final step: Launch Web Application")


if __name__ == '__main__':
    main()