#!/usr/bin/env python3
"""
============================================================
fgsm.py — Fast Gradient Sign Method (FGSM) Attack
Author: Anvesh Raju Vishwaraju
MITRE ATLAS: AML.T0043 - Craft Adversarial Data

Usage: python src/attacks/fgsm.py --epsilon 0.1
============================================================
"""
import torch
import torch.nn as nn
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model import SOCAlertNet, load_model


def fgsm_attack(model, loss_fn, X, y, epsilon):
    """
    Fast Gradient Sign Method attack.
    Adds epsilon * sign(gradient) to input to fool the model.
    """
    X_adv = X.clone().detach().requires_grad_(True)
    output = model(X_adv)
    loss   = loss_fn(output, y)
    model.zero_grad()
    loss.backward()

    X_adv = X_adv + epsilon * X_adv.grad.sign()
    X_adv = torch.clamp(X_adv, 0, 1).detach()
    return X_adv


def evaluate(model, X, y, label=""):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
        acc   = (preds == y).float().mean().item() * 100
    print(f"  {label:<30} Accuracy: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="FGSM Attack")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Perturbation magnitude (0.0-1.0)")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  FGSM ATTACK DEMO")
    print(f"  Author: Anvesh Raju Vishwaraju")
    print(f"  Epsilon: {args.epsilon}")
    print(f"{'='*50}\n")

    # Load model and test data
    if not os.path.exists("models/clean_model.pt"):
        print("[!] Run train_clean.py first: python src/train_clean.py")
        return

    model   = load_model("models/clean_model.pt", SOCAlertNet)
    data    = torch.load("models/test_data.pt")
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    loss_fn = nn.CrossEntropyLoss()

    print("[*] Evaluating clean model...")
    clean_acc = evaluate(model, X_test, y_test, "Clean accuracy")

    print(f"\n[*] Running FGSM attack (epsilon={args.epsilon})...")
    X_adv    = fgsm_attack(model, loss_fn, X_test, y_test, args.epsilon)
    adv_acc  = evaluate(model, X_adv, y_test, "Adversarial accuracy")

    print(f"\n{'='*50}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"  Clean Accuracy:       {clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"  Accuracy Drop:        {clean_acc - adv_acc:.2f}%")
    print(f"  Attack Success Rate:  {100 - adv_acc:.2f}%")
    print(f"\n  Security Insight:")
    print(f"  A SOC ML model with {clean_acc:.0f}% accuracy can be")
    print(f"  reduced to {adv_acc:.0f}% under FGSM attack.")
    print(f"  Run adversarial_training.py to improve robustness.\n")


if __name__ == "__main__":
    main()
