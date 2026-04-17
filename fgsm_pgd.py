#!/usr/bin/env python3
"""
============================================================
fgsm.py — Fast Gradient Sign Method (FGSM) Attack
Author: Anvesh Raju Vishwaraju
MITRE ATLAS: AML.T0043 - Craft Adversarial Data
============================================================
"""
import torch
import torch.nn as nn
import argparse

def fgsm_attack(model: nn.Module,
                loss_fn,
                X: torch.Tensor,
                y: torch.Tensor,
                epsilon: float = 0.1) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    Creates adversarial examples by adding perturbation
    in the direction of the gradient of the loss w.r.t. input.

    Args:
        model:   Target neural network
        loss_fn: Loss function (CrossEntropyLoss)
        X:       Input tensor (batch)
        y:       True labels
        epsilon: Perturbation magnitude (0.0 to 1.0)

    Returns:
        X_adv: Adversarial examples tensor
    """
    X_adv = X.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(X_adv)
    loss = loss_fn(output, y)

    # Backward pass — compute gradients w.r.t. input
    model.zero_grad()
    loss.backward()

    # FGSM perturbation: add epsilon * sign(gradient)
    perturbation = epsilon * X_adv.grad.sign()
    X_adv = X_adv + perturbation

    # Clamp to valid input range [0, 1]
    X_adv = torch.clamp(X_adv, 0, 1).detach()

    return X_adv


def evaluate_attack(model: nn.Module,
                    X_clean: torch.Tensor,
                    X_adv: torch.Tensor,
                    y: torch.Tensor) -> dict:
    """Compare model accuracy on clean vs adversarial inputs."""
    model.eval()
    with torch.no_grad():
        # Clean accuracy
        clean_preds = model(X_clean).argmax(dim=1)
        clean_acc = (clean_preds == y).float().mean().item()

        # Adversarial accuracy
        adv_preds = model(X_adv).argmax(dim=1)
        adv_acc = (adv_preds == y).float().mean().item()

    return {
        "clean_accuracy": round(clean_acc * 100, 2),
        "adversarial_accuracy": round(adv_acc * 100, 2),
        "attack_success_rate": round((1 - adv_acc) * 100, 2),
        "accuracy_drop": round((clean_acc - adv_acc) * 100, 2),
    }


# ============================================================
# pgd.py — Projected Gradient Descent (PGD) Attack
# Stronger iterative version of FGSM
# ============================================================

def pgd_attack(model: nn.Module,
               loss_fn,
               X: torch.Tensor,
               y: torch.Tensor,
               epsilon: float = 0.1,
               alpha: float = 0.01,
               num_steps: int = 40) -> torch.Tensor:
    """
    PGD (Projected Gradient Descent) adversarial attack.

    Stronger than FGSM — applies FGSM iteratively with
    projection back to epsilon-ball after each step.

    Args:
        model:     Target neural network
        loss_fn:   Loss function
        X:         Input tensor
        y:         True labels
        epsilon:   Maximum perturbation magnitude
        alpha:     Step size per iteration
        num_steps: Number of attack iterations

    Returns:
        X_adv: Adversarial examples tensor
    """
    # Start from random point within epsilon-ball
    X_adv = X.clone().detach()
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    X_adv = torch.clamp(X_adv, 0, 1).detach()

    for step in range(num_steps):
        X_adv.requires_grad_(True)

        output = model(X_adv)
        loss = loss_fn(output, y)

        model.zero_grad()
        loss.backward()

        # Step in gradient sign direction
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()

            # Project back to epsilon-ball around original X
            delta = torch.clamp(X_adv - X, -epsilon, epsilon)
            X_adv = torch.clamp(X + delta, 0, 1)

    return X_adv.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Attack Demo")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--steps",   type=int,   default=40)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  ADVERSARIAL ML ATTACK DEMO")
    print(f"  Author: Anvesh Raju Vishwaraju")
    print(f"  Attack: {args.attack.upper()} | Epsilon: {args.epsilon}")
    print(f"{'='*55}\n")
    print("  Run train_clean.py first to generate a model,")
    print("  then run evaluate.py for full results.\n")
