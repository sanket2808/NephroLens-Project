"""
NephroLens - Step 10: Comprehensive Model Evaluation
Author: Sanket Kelzarkar
Description: Evaluate YOLOv8, ResNet50, VM-UNet and Ensemble on the test set.
             Generates confusion matrices, ROC curves, PR curves, and a
             full comparison report.
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)

# ──────────────────────────────────────────────
PROJECT_DIR    = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
RESNET_DIR     = os.path.join(PROJECT_DIR, "resnet_training")
VMUNET_DIR     = os.path.join(PROJECT_DIR, "vmunet_training")
YOLO_DIR       = os.path.join(PROJECT_DIR, "yolov8_training")
EVAL_DIR       = os.path.join(PROJECT_DIR, "evaluation_results")
os.makedirs(EVAL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Stone', 'Normal']
# ──────────────────────────────────────────────


# ═══════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════
class EvalDataset(Dataset):
    def __init__(self, data_dir, split='test', img_size=224):
        self.samples, self.labels = [], []
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
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
        return self.tf(img), self.labels[idx], self.samples[idx]


# ═══════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════
def load_resnet():
    path = os.path.join(RESNET_DIR, 'resnet50_best.pth')
    if not os.path.exists(path):
        print(f"  [SKIP] ResNet weights not found: {path}")
        return None
    m = models.resnet50(pretrained=False)
    m.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(m.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 2)
    )
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m.to(DEVICE)


def load_vmunet():
    """Import & load VM-UNet saved weights."""
    path = os.path.join(VMUNET_DIR, 'vmunet_best.pth')
    if not os.path.exists(path):
        print(f"  [SKIP] VM-UNet weights not found: {path}")
        return None
    try:
        import importlib, sys
        sys.path.insert(0, PROJECT_DIR)
        mod = importlib.import_module('8_model_vmunet_mamba')
        m   = mod.VMUNet(in_channels=3, num_classes=2, img_size=224,
                         embed_dim=128, encoder_depth=3, d_state=8)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        return m.to(DEVICE)
    except Exception as e:
        print(f"  [SKIP] VM-UNet load error: {e}")
        return None


def load_yolo():
    path = os.path.join(YOLO_DIR, 'yolov8n_kidney_stone', 'weights', 'best.pt')
    if not os.path.exists(path):
        print(f"  [SKIP] YOLOv8 weights not found: {path}")
        return None
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except Exception as e:
        print(f"  [SKIP] YOLOv8 load error: {e}")
        return None


# ═══════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════
@torch.no_grad()
def infer_torch(model, loader, is_vmunet=False):
    """Return (probs_class0, true_labels) for any torch model."""
    all_probs, all_labels = [], []
    for imgs, lbls, _ in tqdm(loader, desc='  Inference'):
        imgs = imgs.to(DEVICE)
        if is_vmunet:
            out, _ = model(imgs)
        else:
            out = model(imgs)
        probs = F.softmax(out, dim=1)[:, 0].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(lbls.numpy())
    return np.array(all_probs), np.array(all_labels)


def infer_yolo(yolo_model, dataset):
    """Return (probs_class0, true_labels) for YOLOv8."""
    all_probs, all_labels = [], []
    for _, lbl, img_path in tqdm(dataset, desc='  YOLOv8 Inference'):
        results = yolo_model.predict(img_path, verbose=False)[0]
        if len(results.boxes) > 0:
            cls  = int(results.boxes[0].cls[0])
            conf = float(results.boxes[0].conf[0])
            prob = conf if cls == 0 else (1 - conf)
        else:
            prob = 0.5
        all_probs.append(prob)
        all_labels.append(lbl)
    return np.array(all_probs), np.array(all_labels)


# ═══════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy':  accuracy_score (y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score   (y_true, y_pred, zero_division=0),
        'f1':        f1_score       (y_true, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score  (y_true, y_prob),
        'avg_prec':  average_precision_score(y_true, y_prob),
        'y_pred':    y_pred,
    }


# ═══════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════
def plot_confusion_matrices(results_dict):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1: axes = [axes]

    for ax, (name, res) in zip(axes, results_dict.items()):
        cm = confusion_matrix(res['y_true'], res['metrics']['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=1, linecolor='white')
        ax.set_title(f'{name}\nAcc: {res["metrics"]["accuracy"]*100:.1f}%',
                     fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    fig.suptitle('Confusion Matrices – NephroLens\nPowered by Sanket Kelzarkar',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrices.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices saved")


def plot_roc_curves(results_dict):
    plt.figure(figsize=(8, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for (name, res), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc = res['metrics']['roc_auc']
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves – NephroLens\nPowered by Sanket Kelzarkar',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC curves saved")


def plot_pr_curves(results_dict):
    plt.figure(figsize=(8, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for (name, res), color in zip(results_dict.items(), colors):
        prec, rec, _ = precision_recall_curve(res['y_true'], res['y_prob'])
        ap = res['metrics']['avg_prec']
        plt.plot(rec, prec, lw=2, color=color, label=f'{name} (AP={ap:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves – NephroLens\nPowered by Sanket Kelzarkar',
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ PR curves saved")


def plot_metric_comparison(results_dict):
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names  = list(results_dict.keys())
    values       = {m: [results_dict[n]['metrics'][m] for n in model_names]
                    for m in metric_names}

    x = np.arange(len(model_names))
    width = 0.15
    colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (mn, vals) in enumerate(values.items()):
        offset = (i - len(metric_names)/2) * width
        bars = ax.bar(x + offset, [v*100 for v in vals], width,
                      label=mn.replace('_',' ').title(), color=colors[i], alpha=0.85)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison – NephroLens\nPowered by Sanket Kelzarkar',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'metric_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Metric comparison chart saved")


# ═══════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════
def ensemble_probs(results_dict, method='mean'):
    """Average probabilities from available models."""
    prob_lists = [v['y_prob'] for v in results_dict.values()]
    stacked = np.vstack(prob_lists)
    if method == 'mean':
        return stacked.mean(axis=0)
    if method == 'max':
        return stacked.max(axis=0)
    return stacked.mean(axis=0)


# ═══════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════
def save_report(results_dict):
    lines = []
    sep   = "=" * 70

    lines += [sep,
              "NEPHROLENS – COMPREHENSIVE MODEL EVALUATION REPORT",
              "Powered by Sanket Kelzarkar",
              sep, ""]

    for name, res in results_dict.items():
        m = res['metrics']
        lines += [f"{'─'*50}",
                  f"MODEL: {name}",
                  f"{'─'*50}",
                  f"  Accuracy  : {m['accuracy']*100:.2f}%",
                  f"  Precision : {m['precision']*100:.2f}%",
                  f"  Recall    : {m['recall']*100:.2f}%",
                  f"  F1-Score  : {m['f1']*100:.2f}%",
                  f"  ROC-AUC   : {m['roc_auc']:.4f}",
                  f"  Avg Prec  : {m['avg_prec']:.4f}", ""]

        cr = classification_report(res['y_true'], m['y_pred'],
                                   target_names=CLASS_NAMES)
        lines += ["  Classification Report:", cr, ""]

    lines += [sep, "END OF REPORT", sep]

    report_path = os.path.join(EVAL_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ Full report saved → {report_path}")

    # Also save JSON
    json_data = {name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in res['metrics'].items()
                        if k != 'y_pred'}
                 for name, res in results_dict.items()}
    with open(os.path.join(EVAL_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(json_data, f, indent=4)
    print("✓ Metrics JSON saved")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("NEPHROLENS – MODEL EVALUATION")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

    # Build test loader
    test_ds     = EvalDataset(PREPROCESSED_DIR, 'test', 224)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"\nTest samples: {len(test_ds)}")

    results = {}

    # ── ResNet50 ──────────────────────────────
    print("\n[1] Evaluating ResNet50...")
    resnet = load_resnet()
    if resnet:
        probs, labels = infer_torch(resnet, test_loader)
        results['ResNet50'] = {'y_true': labels, 'y_prob': probs,
                                'metrics': compute_metrics(labels, probs)}
        print(f"  Accuracy: {results['ResNet50']['metrics']['accuracy']*100:.2f}%")

    # ── VM-UNet ───────────────────────────────
    print("\n[2] Evaluating VM-UNet (Mamba SSM)...")
    vmunet = load_vmunet()
    if vmunet:
        probs, labels = infer_torch(vmunet, test_loader, is_vmunet=True)
        results['VM-UNet'] = {'y_true': labels, 'y_prob': probs,
                               'metrics': compute_metrics(labels, probs)}
        print(f"  Accuracy: {results['VM-UNet']['metrics']['accuracy']*100:.2f}%")

    # ── YOLOv8 ───────────────────────────────
    print("\n[3] Evaluating YOLOv8...")
    yolo = load_yolo()
    if yolo:
        probs, labels = infer_yolo(yolo, test_ds)
        results['YOLOv8'] = {'y_true': labels, 'y_prob': probs,
                              'metrics': compute_metrics(labels, probs)}
        print(f"  Accuracy: {results['YOLOv8']['metrics']['accuracy']*100:.2f}%")

    if not results:
        print("\n⚠ No model weights found. Please complete training steps first.")
        return

    # ── Ensemble ──────────────────────────────
    if len(results) > 1:
        print("\n[4] Computing Ensemble...")
        ens_probs = ensemble_probs(results)
        # Use labels from any model (they share the same test set)
        any_labels = next(iter(results.values()))['y_true']
        results['Ensemble'] = {'y_true': any_labels, 'y_prob': ens_probs,
                                'metrics': compute_metrics(any_labels, ens_probs)}
        print(f"  Ensemble Accuracy: {results['Ensemble']['metrics']['accuracy']*100:.2f}%")

    # ── Plots ─────────────────────────────────
    print("\n[5] Generating visualisations...")
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_metric_comparison(results)

    # ── Report ────────────────────────────────
    print("\n[6] Saving report...")
    save_report(results)

    # ── Summary Table ─────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    rows = []
    for name, res in results.items():
        m = res['metrics']
        rows.append({'Model': name,
                     'Accuracy': f"{m['accuracy']*100:.2f}%",
                     'Precision': f"{m['precision']*100:.2f}%",
                     'Recall': f"{m['recall']*100:.2f}%",
                     'F1': f"{m['f1']*100:.2f}%",
                     'AUC': f"{m['roc_auc']:.4f}"})
    df = pd.DataFrame(rows).set_index('Model')
    print(df.to_string())

    print(f"\n✓ All results saved to: {EVAL_DIR}")
    print("=" * 60)
    print("EVALUATION COMPLETED!")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

    # Replace the load_vmunet() function in 10_model_evaluation.py with this:

def load_vmunet():
    """Load VM-UNet using file-path import (fixes numeric-prefix module name issue)."""
    path = os.path.join(VMUNET_DIR, 'vmunet_best.pth')
    if not os.path.exists(path):
        print(f"  [SKIP] VM-UNet weights not found: {path}")
        return None
    try:
        import importlib.util

        # Use spec_from_file_location to avoid the '8_...' numeric prefix import error
        script_path = os.path.join(PROJECT_DIR, '8_model_vmunet.py')
        spec = importlib.util.spec_from_file_location("vmunet_module", script_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        m = mod.VMUNet(in_channels=3, num_classes=2, img_size=224,
                       embed_dim=128, encoder_depth=3, d_state=8)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        return m.to(DEVICE)
    except Exception as e:
        print(f"  [SKIP] VM-UNet load error: {e}")
        return None


if __name__ == '__main__':
    main()
    print("\n✓ Ready for next step: Hyperparameter Tuning (Step 11)")