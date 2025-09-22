import numpy as np

def check_data_validity(x, y) -> None:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: Features has {x.shape[0]} rows, Labels has {y.shape[0]}.")
    try:
        x_fin = np.isfinite(x.astype(float)).all()
    except Exception:
        x_fin = np.isfinite(x).all()
    try:
        y_fin = np.isfinite(y.astype(float)).all()
    except Exception:
        y_fin = np.isfinite(y).all()
    if not (x_fin and y_fin):
        raise ValueError("Found non-finite values (NaN/Inf).")
