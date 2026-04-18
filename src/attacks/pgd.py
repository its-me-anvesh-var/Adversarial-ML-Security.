#!/usr/bin/env python3
"""
============================================================
pgd.py — Projected Gradient Descent (PGD) Attack
Author: Anvesh Raju Vishwaraju
MITRE ATLAS: AML.T0043 - Craft Adversarial Data

PGD is the strongest first-order attack — iterative FGSM
with random start and projection back to epsilon-ball.

Usage: python src/attacks/pgd.py --epsilon 0.1 --steps 40
============================================================
"""
import torch
import torch.nn as nn
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.model import SOCAlertNet, load_model


def pgd_attack(model, loss_fn, X, y, epsilon,
               alpha=0.01, num_steps=40):
    """
    PGD Attack — strongest first-order adversarial attack.

    Steps:
    1. Start from random point within epsilon-ball
    2. Iteratively apply FGSM step of size alpha
    3. Project back to epsilon-ball after each step
    4. Repeat for num_steps iterations
    """
    # Random start within epsilon-ball
    X_adv = X.clone().detach()
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    X_adv = torch.clamp(X_adv, 0, 1).detach()

    for step in range(num_steps):
        X_adv.requires_grad_(True)
        output = model(X_adv)
        loss   = loss_fn(output, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Step in gradient direction
            X_adv = X_adv + alpha * X_adv.grad.sign()
            # Project back to epsilon-ball
            delta = torch.clamp(X_adv - X, -epsilon, epsilon)
            X_adv = torch.clamp(X + delta, 0, 1)

    return X_adv.detach()


def evaluate(model, X, y, label=""):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
        acc   = (preds == y).float().mean().item() * 100
    print(f"  {label:<30} Accuracy: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="PGD Attack")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Max perturbation magnitude")
    parser.add_argument("--alpha",   type=float, default=0.01,
                        help="Step size per iteration")
    parser.add_argument("--steps",   type=int,   default=40,
                        help="Number of attack iterations")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  PGD ATTACK DEMO (Strongest First-Order Attack)")
    print(f"  Author: Anvesh Raju Vishwaraju")
    print(f"  Epsilon: {args.epsilon} | Alpha: {args.alpha} | Steps: {args.steps}")
    print(f"{'='*55}\n")

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

    print(f"\n[*] Running PGD attack ({args.steps} steps)...")
    X_adv   = pgd_attack(model, loss_fn, X_test, y_test,
                          epsilon=args.epsilon, alpha=args.alpha,
                          num_steps=args.steps)
    adv_acc = evaluate(model, X_adv, y_test, "Adversarial accuracy")

    print(f"\n{'='*55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  Clean Accuracy:       {clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"  Accuracy Drop:        {clean_acc - adv_acc:.2f}%")
    print(f"  Attack Success Rate:  {100 - adv_acc:.2f}%")
    print(f"\n  PGD is significantly stronger than FGSM.")
    print(f"  Run adversarial_training.py to defend against this.\n")


if __name__ == "__main__":
    main()
