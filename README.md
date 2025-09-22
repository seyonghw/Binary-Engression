This project implements binary Engression methodology.

## Setup

### Prerequisites
- Python **3.11** recommended (3.10+ OK)
- Git

### 1) Create & Activate a Virtual Environment
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

### 2) Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Usage Example
```
# Binary Engression: learn P(Y=1 | X=x) with Engressor
# ----------------------------------------------------
import torch
import pandas as pd
from engression import engression  # exposes the Engressor via the helper

# 1) Load data (example: Heart Disease; choose ONE continuous x)
df = pd.read_csv("heart.csv")           # <- put your path here
x = torch.tensor(df[["thalach"]].values, dtype=torch.float32)  # 1-D predictor
y = torch.tensor((df["num"] > 0).astype("float32").values).view(-1, 1)  # binary target in {0,1}

# 2) Fit binary engression (sigmoid output; energy-loss training)
model = engression(
    x, y,
    classification=False,        # we treat binary as bounded regression
    out_act="sigmoid",           # force outputs in (0,1)
    hidden_dim=64, noise_dim=32,
    beta=1.0, lr=1e-3, num_epochs=300, batch_size=256,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False
)

# 3) Predict class probability and make decisions
p1 = model.predict(x, target="mean")         # P(Y=1 | X), shape [n,1]
y_hat = (p1 >= 0.5).float()                  # 0/1 prediction at threshold 0.5

# 4) (Optional) Other statistics from the conditional distribution
q10 = model.predict(x, target=0.10)          # 10th percentile of Y|X
q90 = model.predict(x, target=0.90)          # 90th percentile
samples = model.sample(x[:5], sample_size=100)  # draws from Y|X for first 5 points (shape [5,1,100])
```
