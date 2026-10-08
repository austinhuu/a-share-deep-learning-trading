from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def gradient_importance(
    model,
    X_sample: np.ndarray,
    feature_cols: List[str],
    device: str = "cpu",
    save_path=None,
) -> pd.DataFrame:
    model.eval()
    model.to(device)

    x = torch.tensor(
        X_sample,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    probs = model(x)
    score = probs.mean()
    score.backward()

    grads = x.grad.detach().cpu().numpy()
    importance = np.mean(np.abs(grads), axis=(0, 1))

    result = pd.DataFrame({
        "feature": feature_cols,
        "gradient_importance": importance
    }).sort_values("gradient_importance", ascending=False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False, encoding="utf-8-sig")

    return result


def permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    device: str = "cpu",
    n_repeats: int = 3,
    random_state: int = 42,
    save_path=None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    model.eval()
    model.to(device)

    with torch.no_grad():
        base_probs = model(
            torch.tensor(X_val, dtype=torch.float32, device=device)
        ).cpu().numpy()

    try:
        base_auc = roc_auc_score(y_val, base_probs)
    except Exception:
        base_auc = 0.5

    records = []

    for j, feature in enumerate(feature_cols):
        drops = []

        for _ in range(n_repeats):
            X_perm = X_val.copy()

            flat_values = X_perm[:, :, j].reshape(-1).copy()
            rng.shuffle(flat_values)
            X_perm[:, :, j] = flat_values.reshape(X_perm[:, :, j].shape)

            with torch.no_grad():
                perm_probs = model(
                    torch.tensor(X_perm, dtype=torch.float32, device=device)
                ).cpu().numpy()

            try:
                perm_auc = roc_auc_score(y_val, perm_probs)
            except Exception:
                perm_auc = 0.5

            drops.append(base_auc - perm_auc)

        records.append({
            "feature": feature,
            "base_auc": float(base_auc),
            "importance_mean": float(np.mean(drops)),
            "importance_std": float(np.std(drops)),
        })

    result = pd.DataFrame(records).sort_values("importance_mean", ascending=False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False, encoding="utf-8-sig")

    return result


def feature_stability(
    importance_dfs: Dict[str, pd.DataFrame],
    importance_col: str,
    save_path=None,
) -> pd.DataFrame:
    rank_tables = []

    for name, df in importance_dfs.items():
        if df is None or df.empty:
            continue

        temp = df[["feature", importance_col]].copy()
        temp[f"{name}_rank"] = temp[importance_col].rank(ascending=False)
        rank_tables.append(temp[["feature", f"{name}_rank"]])

    if not rank_tables:
        raise ValueError("没有可用的特征重要性结果，无法计算稳定性。")

    merged = rank_tables[0]

    for table in rank_tables[1:]:
        merged = merged.merge(table, on="feature", how="outer")

    rank_cols = [c for c in merged.columns if c.endswith("_rank")]
    merged["rank_mean"] = merged[rank_cols].mean(axis=1)
    merged["rank_std"] = merged[rank_cols].std(axis=1)
    merged = merged.sort_values("rank_mean")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(save_path, index=False, encoding="utf-8-sig")

    return merged


def plot_feature_importance(
    importance_df: pd.DataFrame,
    value_col: str,
    title: str,
    top_n: int = 15,
    save_path=None,
):
    plot_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature"], plot_df[value_col])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close()


def shap_importance(
    model,
    X_background: np.ndarray,
    X_sample: np.ndarray,
    feature_cols: List[str],
    device: str = "cpu",
    save_path=None,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("请先安装 shap：python -m pip install shap") from exc

    model.eval()
    model.to(device)

    class ShapModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            out = self.base_model(x)

            if out.dim() == 1:
                out = out.unsqueeze(1)

            return out

    wrapped_model = ShapModelWrapper(model)
    wrapped_model.eval()
    wrapped_model.to(device)

    background = torch.tensor(X_background, dtype=torch.float32, device=device)
    sample = torch.tensor(X_sample, dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        importance = np.mean(np.abs(shap_values), axis=(0, 1))
    elif shap_values.ndim == 4:
        importance = np.mean(np.abs(shap_values), axis=(0, 1, 3))
    else:
        raise ValueError(f"无法识别的 SHAP 输出维度: {shap_values.shape}")

    result = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": importance
    }).sort_values("shap_importance", ascending=False)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False, encoding="utf-8-sig")

    return result