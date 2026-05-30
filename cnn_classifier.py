#!/usr/bin/env python3
"""
1D ResNet CNN classifier for satellite downlink IQ data
"""
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    precision_recall_curve,
    average_precision_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SAMPLES_DIR = Path("/home/ubuntu/rf_analysis-main/rf_analysis-main/dwingeloo/samples")
ANNOT_PATH = Path("/home/ubuntu/rf_analysis-main/rf_analysis-main/dwingeloo/annot.json")
CLS_MAP_PATH = Path("/home/ubuntu/rf_analysis-main/rf_analysis-main/dwingeloo/cls_map.json")
METRICS_DIR = Path("artifacts_metrics")

BATCH_SIZE = 32
EPOCHS = 2  
LR_MAX = 2e-3
WEIGHT_DECAY = 1e-4
TARGET_LEN = 240_000
MAX_FILES_PER_LABEL = 1000

DIRECTORY_LABEL_OVERRIDES = {
    "APRIZESAT_DIR": "APRIZESAT", 
    "NOAA_DIR": "NOAA", 
    "CONTECSAT-1_DIR": "CONTECSAT-1",
    "BOTSAT_DIR": "BOTSAT", 
    "BLUEBON_DIR": "BLUEBON", 
    "OTTER_PUP2_DIR": "OTTER",
    "BRO_13_DIR": "BRO-13", 
    "HUBBLE_7_DIR": "HUBBLE-7", 
    "STARLINK_DIR": "STARLINK_BEACON",
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_json(path: Path) -> dict:
    """
    Load and parse a JSON file from disk.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_2x(iq: np.ndarray) -> np.ndarray:
    """
    Ensure the IQ array conforms to a 2-row format where row 0 is I and row 1 is Q.

    Handles 1D flattened arrays and transposition of N x 2 arrays.
    """
    if iq.ndim == 2:
        if iq.shape[0] == 2: 
            return iq
        if iq.shape[1] == 2: 
            return iq.T
    if iq.ndim == 1:
        if iq.size % 2 != 0:
            iq = iq[:-1]
        return iq.reshape(-1, 2).T
    raise ValueError(f"Bad IQ shape: {iq.shape}")


def center_crop_or_pad_2x(iq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Enforce a static length across all input signals via center cropping or zero padding.
    """
    n = iq.shape[1]
    if n > target_len:
        start = (n - target_len) // 2
        return iq[:, start:start + target_len]
    if n < target_len:
        pad = target_len - n
        return np.pad(iq, ((0, 0), (pad // 2, pad - pad // 2)))
    return iq


def build_iq_channels(iq_2x: np.ndarray) -> np.ndarray:
    """
    Normalize IQ signals to zero mean and unit variance for training stability.
    """
    out = iq_2x.astype(np.float32, copy=False)
    mean_val = out.mean()
    std_val = out.std()
    return (out - mean_val) / (std_val + 1e-6)


def augment_iq(iq_2x: np.ndarray) -> np.ndarray:
    """
    Apply RF-domain data augmentations to synthetic/real signal samples.

    Includes amplitude scaling, time-domain shifting, carrier frequency offsets
    via complex rotation, and additive white Gaussian noise.
    """
    i = iq_2x[0].copy()
    q = iq_2x[1].copy()

    gain = np.exp(np.random.uniform(np.log(0.7), np.log(1.3)))
    i = i * gain
    q = q * gain

    if np.random.rand() < 0.8:
        shift = np.random.randint(-4000, 4001)
        i = np.roll(i, shift)
        q = np.roll(q, shift)

    if np.random.rand() < 0.6:
        freq = np.random.uniform(-0.0025, 0.0025)
        phase = 2.0 * np.pi * freq * np.arange(i.shape[0])
        c = np.cos(phase)
        s = np.sin(phase)
        new_i = i * c - q * s
        new_q = i * s + q * c
        i = new_i
        q = new_q

    if np.random.rand() < 0.9:
        noise_std = np.random.uniform(0.005, 0.03)
        i = i + np.random.normal(0.0, noise_std, size=i.shape)
        q = q + np.random.normal(0.0, noise_std, size=q.shape)

    return np.stack([i, q], axis=0).astype(np.float32)


class IQDataset(Dataset):
    def __init__(self, items, train=False):
        """
        Initialize the IQ signal dataset structure.
        """
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        raw_arr = np.load(path)
        iq = ensure_2x(raw_arr)
        iq = center_crop_or_pad_2x(iq, TARGET_LEN)

        if self.train:
            iq = augment_iq(iq)

        x = build_iq_channels(iq)
        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.long)


class BasicBlock1D(nn.Module):
    """
    Residual block containing 1D CNN components for handling time-series RF sequences.
    """
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        """
        Set up the layers of the residual building block.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut_x = self.shortcut(x)
        return F.relu(out + shortcut_x)


class ResNet1D(nn.Module):
    """
    A 1D ResNet Architecture designed for analyzing raw temporal patterns from IQ data.
    """
    def __init__(self, num_classes):
        """
        Initialize network architecture layers.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.layer1 = nn.Sequential(
            BasicBlock1D(64, 64, stride=1, dropout=0.1), 
            BasicBlock1D(64, 64, stride=1, dropout=0.1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock1D(64, 128, stride=2, dropout=0.1), 
            BasicBlock1D(128, 128, stride=1, dropout=0.1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock1D(128, 256, stride=2, dropout=0.1), 
            BasicBlock1D(256, 256, stride=1, dropout=0.1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock1D(256, 384, stride=2, dropout=0.15), 
            BasicBlock1D(384, 384, stride=1, dropout=0.15)
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(384, 256), 
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run evaluation metrics and calculate cross-entropy loss across a targeted data loader.
    """
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    y_prob = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        loss_val = F.cross_entropy(logits, y, label_smoothing=0.05).item()
        losses.append(loss_val)
        y_true.append(y.cpu())
        y_pred.append(probs.argmax(dim=1).cpu())
        y_prob.append(probs.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_prob = torch.cat(y_prob).numpy()
    
    mean_loss = float(np.mean(losses))
    accuracy = float((y_true == y_pred).mean())
    return mean_loss, accuracy, y_true, y_pred, y_prob


def get_label(key, meta):
    """
    Parse out matching metadata descriptors to create uniform class labels.
    """
    if key in DIRECTORY_LABEL_OVERRIDES:
        return DIRECTORY_LABEL_OVERRIDES[key].strip().upper()
    sat_name = meta.get("Satellite", "")
    if sat_name:
        return str(sat_name).strip().upper()
    if key.endswith("_DIR"):
        clean_key = key[:-4]
    else:
        clean_key = key
    return clean_key.replace("_", "-").upper()


def build_items_from_annot_mixed(annot_path, samples_dir, cls_map_path):
    """
    Crawl through input locations to resolve directory pathways and metadata records.
    """
    annotations = load_json(annot_path)
    cls_map = load_json(cls_map_path)
    inv_cls_map = {}
    for k, v in cls_map.items():
        inv_cls_map[int(v)] = k
        
    paths_by_label = defaultdict(list)
    seen_paths = set()

    for key, meta in annotations.items():
        dir_name = meta.get("dir", "").strip()
        if dir_name:
            label = get_label(key, meta)
            folder_path = Path(dir_name).expanduser()
            for p in sorted(folder_path.glob("*.npy")):
                resolved_str = str(p.resolve())
                if resolved_str not in seen_paths:
                    seen_paths.add(resolved_str)
                    paths_by_label[label].append(p)

        orig_path = samples_dir / f"{key}.npy"
        sat_name = meta.get("Satellite", "").strip()
        if orig_path.exists():
            if sat_name in cls_map:
                class_idx = int(cls_map[sat_name])
                label = str(inv_cls_map[class_idx]).strip().upper()
                resolved_str = str(orig_path.resolve())
                if resolved_str not in seen_paths:
                    seen_paths.add(resolved_str)
                    paths_by_label[label].append(orig_path)

    label_names = sorted(paths_by_label.keys())
    label_to_id = {}
    for i, name in enumerate(label_names):
        label_to_id[name] = i

    items = []
    for label in label_names:
        files = paths_by_label[label]
        random.shuffle(files)
        selected_files = files[:MAX_FILES_PER_LABEL]
        for path in selected_files:
            items.append((str(path), label_to_id[label]))

    random.shuffle(items)
    return items, label_names


def main():
    """
    Main driver method orchestration structure. Handles resource setup,
    the active training iteration steps, evaluation cycles, and metrics logging outputs.
    """
    json_path = METRICS_DIR / "metrics_summary.json"
    best_path = METRICS_DIR / "best_model.pt"
    
    data_all, label_names = build_items_from_annot_mixed(ANNOT_PATH, SAMPLES_DIR, CLS_MAP_PATH)
    print("Labels counts:", len(label_names))
    print("Total samples raw:", len(data_all))

    data = []
    for path, y in tqdm(data_all, desc="Validating files"):
        try:
            arr = np.load(path, mmap_mode="r")
            ensure_2x(np.asarray(arr))
            data.append((path, y))
        except Exception:
            continue

    if not data: 
        raise ValueError("No valid samples left.")

    y_all = []
    for path, y in data:
        y_all.append(y)
        
    counts = Counter(y_all)
    can_stratify = False
    if len(counts) > 1:
        if min(counts.values()) >= 2:
            can_stratify = True
    
    if can_stratify:
        strat_param = y_all
    else:
        strat_param = None
        
    train, test = train_test_split(
        data, test_size=0.2, random_state=SEED, stratify=strat_param
    )

    train_dataset = IQDataset(train, train=True)
    test_dataset = IQDataset(test, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Running on device:", device)

    model = ResNet1D(num_classes=len(label_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.15
    )
    
    is_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    best_acc = -1.0

    print("Training Loop:")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=is_cuda):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
                
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            running_loss = running_loss + loss.item()

        val_loss, val_acc, _, _, _ = evaluate(model, test_loader, device)
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} | Loss: {avg_train_loss} | Val Loss: {val_loss} | Val Acc: {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print("Generating Metrics & Plots")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        
    val_loss, val_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    all_class_ids = list(range(len(label_names)))

    macro_precision = precision_score(y_true, y_pred, labels=all_class_ids, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=all_class_ids, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, labels=all_class_ids, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=all_class_ids, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=all_class_ids, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=all_class_ids, average="weighted", zero_division=0)

    topk = np.argsort(-y_prob, axis=1)[:, :2]
    
    match_count = 0
    for yt, row in zip(y_true, topk):
        if yt in row:
            match_count = match_count + 1
    top2 = match_count / len(y_true)

    report = classification_report(y_true, y_pred, labels=all_class_ids, target_names=label_names, zero_division=0, output_dict=True)

    per_class = {}
    for class_id, label in enumerate(label_names):
        row = report[label]
        per_class[label] = {
            "precision": float(row["precision"]), 
            "recall": float(row["recall"]),
            "f1_score": float(row["f1-score"]), 
            "support": int(row["support"]),
        }

    print(f"Test Metrics -> Acc: {val_acc} | Top-2: {top2} | Val Loss: {val_loss}")

    fig, ax = plt.subplots(figsize=(14, 10))
    cm = confusion_matrix(y_true, y_pred, labels=all_class_ids)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(cmap="Blues", values_format="d", xticks_rotation=45, ax=ax)
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    y_bin = label_binarize(y_true, classes=all_class_ids)
    average_precision_by_class = {}

    for class_id, label in enumerate(label_names):
        if y_bin[:, class_id].sum() == 0:
            average_precision_by_class[label] = None
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, class_id], y_prob[:, class_id])
        average_precision = average_precision_score(y_bin[:, class_id], y_prob[:, class_id])
        average_precision_by_class[label] = float(average_precision)
        ax.plot(recall, precision, label=f"{label} AP={average_precision}")

    ax.set_title("PR Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    summary_data = {
        "overall": {
            "accuracy": float(val_acc), 
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall), 
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall), 
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1), 
            "top2_accuracy": float(top2), 
            "val_loss": float(val_loss),
        },
        "per_class": per_class,
        "average_precision_by_class": average_precision_by_class,
        "label_names": label_names,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)


if __name__ == "__main__":
    main()
