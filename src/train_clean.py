#!/usr/bin/env python3
"""
============================================================
train_clean.py — Train Baseline (Clean) Model
Author: Anvesh Raju Vishwaraju

Trains a standard neural network on synthetic SOC alert data.
This baseline model will then be attacked by FGSM and PGD
to demonstrate adversarial vulnerability.
============================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import SOCAlertNet, save_model

torch.manual_seed(42)
np.random.seed(42)

# ── Synthetic Dataset ─────────────────────────────────────────

def generate_soc_data(n: int = 10000):
    """
    Generate synthetic SOC alert dataset.
    Features: severity, ip_reputation, alert_freq, bytes,
              hour, is_admin, failed_logins, alert_type
    Label: 0=False Positive, 1=True Positive
    """
    X = np.random.rand(n, 8).astype(np.float32)

    # Scale features to realistic ranges
    X[:, 0] = (X[:, 0] * 3 + 1).astype(int)  # severity 1-4
    X[:, 1] = X[:, 1] * 100                   # ip_reputation 0-100
    X[:, 2] = X[:, 2] * 99 + 1                # alert_frequency 1-100
    X[:, 3] = X[:, 3] * 1e7                   # bytes 0-10M
    X[:, 4] = (X[:, 4] * 23).astype(int)      # hour 0-23
    X[:, 5] = (X[:, 5] > 0.8).astype(float)   # is_admin 20% True
    X[:, 6] = X[:, 6] * 50                    # failed_logins 0-50
    X[:, 7] = (X[:, 7] * 4).astype(int)       # alert_type 0-4

    # Normalize to [0,1]
    X_norm = X.copy()
    X_norm[:, 0] /= 4
    X_norm[:, 1] /= 100
    X_norm[:, 2] /= 100
    X_norm[:, 3] /= 1e7
    X_norm[:, 4] /= 23
    X_norm[:, 6] /= 50
    X_norm[:, 7] /= 4

    # Generate labels (realistic ~15% TP rate)
    score = (
        X_norm[:, 0] * 0.30 +   # severity
        X_norm[:, 1] * 0.25 +   # ip reputation
        X_norm[:, 5] * 0.20 +   # is_admin
        X_norm[:, 6] * 0.15 +   # failed_logins
        ((X_norm[:, 4] < 0.35) | (X_norm[:, 4] > 0.87)) * 0.10  # after-hours
    )
    score += np.random.normal(0, 0.08, n)
    y = (score > 0.62).astype(int)

    return torch.FloatTensor(X_norm), torch.LongTensor(y)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (output.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / len(loader), correct / total


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        output = model(X.to(device))
        preds  = output.argmax(1).cpu()
        acc    = (preds == y).float().mean().item()
    return acc


def main():
    print("\n" + "="*50)
    print("  ADVERSARIAL ML — Clean Model Training")
    print("  Author: Anvesh Raju Vishwaraju")
    print("="*50 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # Generate data
    print("[*] Generating synthetic SOC alert dataset...")
    X, y = generate_soc_data(n=10000)
    n_train = 8000
    X_train, y_train = X[:n_train], y[:n_train]
    X_test,  y_test  = X[n_train:], y[n_train:]
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Class balance — TP: {y.sum().item()} ({y.float().mean().item()*100:.1f}%)")

    # DataLoader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Model
    model   = SOCAlertNet(input_dim=8, num_classes=2).to(device)
    opt     = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    sched   = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    # Training loop
    print("\n[*] Training baseline model...")
    print(f"{'Epoch':>6} {'Loss':>8} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 40)

    best_acc = 0
    for epoch in range(1, 31):
        loss, train_acc = train_epoch(model, loader, opt, loss_fn, device)
        test_acc        = evaluate(model, X_test, y_test, device)
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} {loss:>8.4f} {train_acc*100:>9.2f}% {test_acc*100:>9.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, "models/clean_model.pt")

    print(f"\n[✓] Best test accuracy: {best_acc*100:.2f}%")
    print("[✓] Model saved to models/clean_model.pt")
    print("\n    Next step: Run attacks to test adversarial robustness")
    print("    python src/attacks/fgsm.py --epsilon 0.1")
    print("    python src/attacks/pgd.py  --epsilon 0.1 --steps 40\n")

    # Save test data for attack scripts
    torch.save({"X_test": X_test, "y_test": y_test}, "models/test_data.pt")


if __name__ == "__main__":
    main()
