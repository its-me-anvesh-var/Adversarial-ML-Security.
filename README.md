# ⚔️ Adversarial ML Security

Implementation and analysis of adversarial machine learning attacks — FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent) — against neural network classifiers, with adversarial training as a defence mechanism. Focused on security implications for ML-based SOC tools.

---

## 📂 Repository Structure

```
adversarial-ml-security/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py                  # Neural network model definition
│   ├── train_clean.py            # Train clean (baseline) model
│   ├── attacks/
│   │   ├── fgsm.py               # FGSM attack implementation
│   │   └── pgd.py                # PGD attack implementation
│   ├── defence/
│   │   └── adversarial_training.py  # Adversarial training defence
│   └── evaluate.py               # Attack success rate evaluation
└── results/
    └── results-summary.md        # Attack vs Defence comparison table
```

---

## 🧪 What This Demonstrates

| Scenario | Attack | Accuracy Before | Accuracy After |
|---|---|---|---|
| Baseline model | FGSM (ε=0.1) | 94.2% | 31.7% |
| Baseline model | PGD (ε=0.1, steps=40) | 94.2% | 11.3% |
| Adversarially trained | FGSM (ε=0.1) | 91.8% | 78.4% |
| Adversarially trained | PGD (ε=0.1, steps=40) | 91.8% | 71.2% |

**Key insight:** A SOC ML model with 94% accuracy can be reduced to 11% under PGD attack — adversarial training restores robustness to 71%.

---

## 🚀 Quick Start

```bash
git clone https://github.com/its-me-anvesh-var/adversarial-ml-security
cd adversarial-ml-security
pip install -r requirements.txt

# Train baseline model
python src/train_clean.py

# Run FGSM attack
python src/attacks/fgsm.py --epsilon 0.1

# Run PGD attack
python src/attacks/pgd.py --epsilon 0.1 --steps 40

# Apply adversarial training defence
python src/defence/adversarial_training.py

# Full evaluation
python src/evaluate.py
```

---

## 🔐 Security Implications

### Why This Matters for SOC/ML Security
1. **Alert classifiers** can be fooled by adversarially crafted log entries
2. **Malware classifiers** can be evaded by slightly perturbing binary features
3. **IDS/IPS ML models** are vulnerable if attackers know the model architecture
4. **Defence:** Adversarial training + input validation + model ensembles

---

## 🏅 Author

**Anvesh Raju Vishwaraju**
M.S. Cybersecurity — UNC Charlotte | M.Tech AI — Univ. of Hyderabad
🔗 [LinkedIn](https://linkedin.com/in/arv007)
