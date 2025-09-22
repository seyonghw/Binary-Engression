import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from engression import engression
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    roc_curve, precision_recall_curve, average_precision_score
)

# Ensure results dir exists at project root
results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(exist_ok=True)

# --- Synthetic data as in the first test ---
def make_easy_logistic(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    eta = 1.0 + 3.0 * x
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(np.float32)
    return x.astype(np.float32), y.astype(int), p.ravel()

x, y, p_true = make_easy_logistic()

# --- Fit engression model and predict ---
m = engression(x, y, classification=True)
p_hat = np.ravel(np.array(m.predict(x)))

# --- Metrics ---
auc = roc_auc_score(y, p_hat)
ll = log_loss(y, p_hat, labels=[0, 1])
acc = accuracy_score(y, (p_hat >= 0.5).astype(int))
base = y.mean()
base_ll = log_loss(y, np.full_like(p_hat, base), labels=[0, 1])
ap = average_precision_score(y, p_hat)

print(f"AUC={auc:.3f}, LogLoss={ll:.3f}, Acc={acc:.3f}, AP={ap:.3f}")

# --- Plot 1: ROC curve ---
fpr, tpr, _ = roc_curve(y, p_hat)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig(results_dir / "roc_curve.png", dpi=160, bbox_inches="tight")
plt.close()

# --- Plot 2: Precision-Recall curve ---
prec, rec, _ = precision_recall_curve(y, p_hat)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.savefig(results_dir / "pr_curve.png", dpi=160, bbox_inches="tight")
plt.close()

# --- Plot 3: Calibration (reliability) diagram ---
def calibration_curve_points(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, cnts = [], [], []
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.any():
            xs.append(y_prob[m].mean())
            ys.append(y_true[m].mean())
            cnts.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(cnts)

cx, cy, cc = calibration_curve_points(y, p_hat, n_bins=15)
plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--")
plt.scatter(cx, cy, s=np.clip(cc, 10, 200))
plt.xlabel("Predicted probability")
plt.ylabel("Empirical frequency")
plt.title("Calibration (Reliability) Diagram")
plt.savefig(results_dir / "calibration_reliability.png", dpi=160, bbox_inches="tight")
plt.close()

# --- Plot 4: Histogram of predicted probabilities ---
plt.figure()
plt.hist(p_hat, bins=30)
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Distribution of Predicted Probabilities")
plt.savefig(results_dir / "pred_prob_hist.png", dpi=160, bbox_inches="tight")
plt.close()
