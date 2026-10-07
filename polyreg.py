import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse, r2_score

warnings.filterwarnings("ignore")
ROLL = "BT2024073"

CONFIG = {
    1: dict(degree=5,  scale=False, make=lambda: Lasso(alpha=0.0025, max_iter=500000)),
    2: dict(degree=10, scale=True,  make=lambda: Ridge(alpha=0.5)),
}


def path(var, split):
    for folder in (".", "data"):
        f = os.path.join(folder, f"{ROLL}_{split}_var{var}.csv")
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"{ROLL}_{split}_var{var}.csv not found in . or ./data")


def load(var, split):
    return pd.read_csv(path(var, split))


def build(var):
    c = CONFIG[var]
    steps = [PolynomialFeatures(c["degree"], include_bias=False)]
    if c["scale"]:
        steps.append(StandardScaler())
    return make_pipeline(*steps, c["make"]())


def cv_pred(make, d, X, y, scale=False, seed=1):
    out = np.zeros(len(y))
    for t, v in KFold(5, shuffle=True, random_state=seed).split(X):
        st = [PolynomialFeatures(d, include_bias=False)] + ([StandardScaler()] if scale else [])
        out[v] = make_pipeline(*st, make()).fit(X[t], y[t]).predict(X[v])
    return out


def evaluate():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    tr = load(1, "train"); X1, y1 = tr.drop(columns="y").values, tr["y"].values
    degs1 = [2, 3, 4, 5, 6, 7]
    lasso1 = [mse(y1, cv_pred(lambda: Lasso(alpha=0.0025, max_iter=500000), d, X1, y1)) for d in degs1]
    ridge1 = [mse(y1, cv_pred(lambda: Ridge(alpha=10), d, X1, y1, scale=True)) for d in degs1]
    va = (np.abs(X1) == 1).sum(1) >= 3
    shift1 = []
    for d in degs1:
        m = make_pipeline(PolynomialFeatures(d, include_bias=False),
                          Lasso(alpha=0.0025, max_iter=500000)).fit(X1[~va], y1[~va])
        shift1.append(mse(y1[va], m.predict(X1[va])))
    print("var1 degrees   :", degs1)
    print("var1 lasso CV  :", np.round(lasso1, 3))
    print("var1 ridge CV  :", np.round(ridge1, 3))
    print("var1 shift val :", np.round(shift1, 3))
    p = cv_pred(CONFIG[1]["make"], 5, X1, y1)
    print("var1 final CV MSE %.4f  R2 %.4f" % (mse(y1, p), r2_score(y1, p)))

    tr = load(2, "train"); X2, y2 = tr.drop(columns="y").values, tr["y"].values
    degs2 = [3, 5, 7, 8, 9, 10, 11, 12, 14, 16, 20]
    ridge2 = [mse(y2, cv_pred(lambda: Ridge(alpha=0.5), d, X2, y2, scale=True)) for d in degs2]
    print("var2 degrees   :", degs2)
    print("var2 ridge CV  :", np.round(ridge2, 4))
    p = cv_pred(CONFIG[2]["make"], 10, X2, y2, scale=True)
    print("var2 final CV MSE %.4f  R2 %.4f" % (mse(y2, p), r2_score(y2, p)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(degs1, lasso1, "o-", label="Lasso, 5-fold CV")
    ax[0].plot(degs1, ridge1, "s--", label="Ridge, 5-fold CV")
    ax[0].plot(degs1, shift1, "^-", label="Lasso, shift-aware val")
    ax[0].set_yscale("log"); ax[0].set(xlabel="polynomial degree", ylabel="MSE", title="var1"); ax[0].legend(fontsize=8)
    ax[1].plot(degs2, ridge2, "o-"); ax[1].set_yscale("log")
    ax[1].set(xlabel="polynomial degree", ylabel="CV MSE", title="var2 (Ridge)")
    plt.tight_layout(); plt.savefig("figures/degree_sweep.png", dpi=160); plt.close()

    te = load(1, "test").values
    fig, ax = plt.subplots(figsize=(5, 3.2)); b = np.arange(-.5, 7.5)
    ax.hist((np.abs(X1) == 1).sum(1), bins=b, alpha=.6, density=True, label="train")
    ax.hist((np.abs(te) == 1).sum(1), bins=b, alpha=.6, density=True, label="test")
    ax.set(xlabel="# of the 6 features sitting at +-1", ylabel="density", title="var1 covariate shift"); ax.legend()
    plt.tight_layout(); plt.savefig("figures/shift.png", dpi=160); plt.close()


def train_and_predict():
    for var in (1, 2):
        tr = load(var, "train"); X, y = tr.drop(columns="y").values, tr["y"].values
        model = build(var).fit(X, y)
        te = load(var, "test")
        out = f"{ROLL}_pred_var{var}.csv"
        pd.DataFrame({"y": model.predict(te.values)}).to_csv(out, index=False)
        print(f"var{var}: trained on {X.shape}, wrote {len(te)} predictions -> {out}")


if __name__ == "__main__":
    if "--eval" in sys.argv:
        evaluate()
    train_and_predict()
