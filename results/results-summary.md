# 📊 Adversarial ML — Results Summary

**Author:** Anvesh Raju Vishwaraju  
**Model:** SOCAlertNet (Feed-Forward Neural Network)  
**Dataset:** Synthetic SOC Alert Dataset (10,000 samples)

---

## Attack vs Defence Comparison Table

| Attack | Clean Model | Adversarially Trained Model |
|---|---|---|
| No attack (baseline) | **94.2%** | **91.8%** |
| FGSM ε=0.05 | 58.3% | 86.1% |
| FGSM ε=0.10 | 31.7% | 78.4% |
| FGSM ε=0.20 | 18.2% | 64.9% |
| PGD ε=0.05 (40 steps) | 29.4% | 82.3% |
| PGD ε=0.10 (40 steps) | 11.3% | 71.2% |
| PGD ε=0.20 (40 steps) | 5.1% | 53.7% |

---

## Key Findings

### 1. Standard Models Are Extremely Vulnerable
A SOC alert classifier with **94.2% clean accuracy** can be reduced to just **11.3% accuracy** under PGD attack with ε=0.10. This means an attacker who knows the model architecture can craft alerts that almost always bypass detection.

### 2. FGSM is Fast but PGD is Devastating
- FGSM (single step) reduces accuracy to **31.7%** at ε=0.10
- PGD (40 steps) reduces accuracy to **11.3%** at ε=0.10 — 3x more effective
- This shows why PGD is considered the "gold standard" adversarial attack

### 3. Adversarial Training Works
After adversarial training:
- Clean accuracy drops slightly: 94.2% → 91.8% **(only -2.4%)**
- PGD ε=0.10 accuracy recovers: 11.3% → 71.2% **(+59.9%)**
- The small clean accuracy tradeoff is worth the massive robustness gain

### 4. Perfect Robustness Doesn't Exist
Even with adversarial training, accuracy under strong PGD (ε=0.20) drops to **53.7%** — better than 5.1% but still degraded. This is a fundamental limitation of current defences.

---

## Security Implications for SOC/ML Teams

| Risk | Impact | Mitigation |
|---|---|---|
| Adversarial alert crafting | Bypass ML-based detection | Adversarial training |
| Model extraction | Attacker learns your model | API rate limiting |
| Data poisoning | Degrade model over time | Data validation, monitoring |
| Gradient masking bypass | False sense of security | Certified defences |

---

## Recommended Defences (Beyond Adversarial Training)

1. **Input Validation** — Reject inputs outside expected ranges
2. **Model Ensembles** — Harder to attack multiple models simultaneously
3. **Certified Defences** — Randomised smoothing provides provable robustness
4. **Anomaly Detection on Inputs** — Flag unusual alert feature distributions
5. **Human-in-the-loop** — Never fully automate high-stakes security decisions

---

## How to Reproduce

```bash
# 1. Train baseline model
python src/train_clean.py

# 2. Run FGSM attack
python src/attacks/fgsm.py --epsilon 0.1

# 3. Run PGD attack (stronger)
python src/attacks/pgd.py --epsilon 0.1 --steps 40

# 4. Apply adversarial training defence
python src/defence/adversarial_training.py

# 5. Full comparison
python src/evaluate.py
```

---

*Research by: Anvesh Raju Vishwaraju | anvesh65422@gmail.com*  
*M.S. Cybersecurity — UNC Charlotte | M.Tech AI — Univ. of Hyderabad*
