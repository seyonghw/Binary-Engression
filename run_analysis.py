import argparse
import sys
from pathlib import Path
from xml.parsers.expat import model
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Import the installed engression package + the test helper
from engression import engression
from tests.test_data_validity import check_data_validity


def main():
    parser = argparse.ArgumentParser(description="Run binary engression analysis and save outputs.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "processed.csv"))
    parser.add_argument("--feature", default="thalach")
    parser.add_argument("--target", default="num")
    parser.add_argument("--out", default=str(RESULTS_DIR / "prob_vs_thalach.png"))
    parser.add_argument("--pred_csv", default=str(RESULTS_DIR / "predictions.csv"))
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000, help="MC samples for probability")
    parser.add_argument("--eval_points", type=int, default=100, help="x-eval grid size")
    args = parser.parse_args()

    # 1) Load data
    df = pd.read_csv(args.input)
    if args.feature not in df.columns or args.target not in df.columns:
        raise ValueError(f"Columns not found. Got: {list(df.columns)}")

    # Extract to NumPy and validate using the shared test helper
    x_np = df[args.feature].to_numpy()
    y_np = df[args.target].to_numpy()
    check_data_validity(x_np, y_np)
    print("[OK] Data validity checks passed.")

    # Device + tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x_np, dtype=torch.float32).view(-1, 1).to(device)
    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(device)

    # 2) Fit engressor (installed package)
    model = engression(
        x, y,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device
    )

    # Evaluation grid
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    x_eval_range = (x_min - 10.0, x_max + 10.0)
    x_eval = torch.linspace(x_eval_range[0], x_eval_range[1], steps=args.eval_points).view(-1, 1).to(device)

    # --- Predict prob via sampling + 0.5 threshold (ensure 1-D outputs) ---
    with torch.no_grad():
        y_sample = model.sample(x_eval, sample_size=args.samples)  # may be torch.Tensor or np.ndarray

        # Convert to torch on the right device
        y_sample_t = torch.as_tensor(y_sample, device=x_eval.device)

        # y_sample_t can be [n_eval, samples] or [n_eval, samples, 1]
        if y_sample_t.ndim == 3:
            # average over sample dim and last singleton dim
            probs_t = (y_sample_t > 0.5).float().mean(dim=(1, 2))
        elif y_sample_t.ndim == 2:
            # average over sample dim
            probs_t = (y_sample_t > 0.5).float().mean(dim=1)
        elif y_sample_t.ndim == 1:
            # already [n_eval]
            probs_t = (y_sample_t > 0.5).float()
        else:
            raise ValueError(f"Unexpected y_sample shape: {tuple(y_sample_t.shape)}")

        y_prob = probs_t.detach().cpu().numpy().reshape(-1)  # 1-D

    # x_eval to 1-D
    x_eval_1d = x_eval.view(-1).detach().cpu().numpy()

    # --- Save predictions table (strictly 1-D columns) ---
    pred_df = pd.DataFrame({
        args.feature: x_eval_1d,
        "pred_prob": y_prob
    })
    pred_df.to_csv(args.pred_csv, index=False)  

    # Plot (save to results)
    plt.figure(figsize=(6, 4))
    plt.scatter(pred_df[args.feature], pred_df["pred_prob"], s=8, label="Predicted P(Y=1|X)")
    plt.ylim(-0.1, 1.1)
    plt.xlabel(args.feature)
    plt.ylabel("Probability (num > 0.5)")
    plt.title("Predicted Probability vs. maximum heart rate achieved")
    plt.legend()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close()

    # Log
    (RESULTS_DIR / "run_log.txt").write_text(
        f"Range of x: ({x_min:.4f}, {x_max:.4f})\n"
        f"Evaluation range: ({x_eval_range[0]:.4f}, {x_eval_range[1]:.4f})\n"
        f"Figure: {args.out}\nPredictions: {args.pred_csv}\n",
        encoding="utf-8"
    )

    print(f"[OK] Saved figure to {args.out}")
    print(f"[OK] Saved predictions to {args.pred_csv}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
