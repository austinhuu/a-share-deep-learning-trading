import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class WeightedEnsemble:
    """
    多模型加权集成。

    支持：
    1. 平均集成 average
    2. AUC线性加权 auc
    3. AUC softmax 加权 softmax_auc

    输入模型必须满足：
        input:  (batch, window, n_features)
        output: (batch,)
    """

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        weights: Optional[Dict[str, float]] = None,
        device: str = "cpu",
    ):
        """
        models:
            {
                "lstm": lstm_model,
                "hybrid": hybrid_model,
                "cnn": cnn_model,
                "mlp": mlp_model
            }

        weights:
            {
                "lstm": 0.3,
                "hybrid": 0.4,
                "cnn": 0.2,
                "mlp": 0.1
            }
        """
        if not models:
            raise ValueError("models 不能为空")

        self.models = models
        self.device = device

        if weights is None:
            self.weights = self._equal_weights(list(models.keys()))
        else:
            self.weights = self._normalize_weights(weights)

        self.to(device)
        self.eval()

    @staticmethod
    def _equal_weights(model_names: List[str]) -> Dict[str, float]:
        weight = 1.0 / len(model_names)
        return {name: weight for name in model_names}

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        clean_weights = {}

        for name, value in weights.items():
            value = float(value)
            if value < 0:
                value = 0.0
            clean_weights[name] = value

        total = sum(clean_weights.values())

        if total <= 1e-12:
            n = len(clean_weights)
            return {name: 1.0 / n for name in clean_weights}

        return {name: value / total for name, value in clean_weights.items()}

    def to(self, device: str):
        self.device = device
        for model in self.models.values():
            model.to(device)
        return self

    def eval(self):
        for model in self.models.values():
            model.eval()
        return self

    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 ensemble 预测概率。

        x:
            torch.Tensor, shape = (batch, window, n_features)

        return:
            torch.Tensor, shape = (batch,)
        """
        x = x.to(self.device)

        weighted_preds = []

        with torch.no_grad():
            for name, model in self.models.items():
                pred = model(x)

                if pred.dim() > 1:
                    pred = pred.squeeze(-1)

                weight = self.weights.get(name, 0.0)
                weighted_preds.append(weight * pred)

        if not weighted_preds:
            raise ValueError("没有可用模型参与集成预测")

        ensemble_pred = torch.stack(weighted_preds, dim=0).sum(dim=0)
        return ensemble_pred

    def predict_numpy(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        numpy 输入版本，方便和你现有回测代码衔接。

        X:
            np.ndarray, shape = (N, window, n_features)

        return:
            np.ndarray, shape = (N,)
        """
        preds = []

        for start in range(0, len(X), batch_size):
            end = start + batch_size
            batch = torch.tensor(X[start:end], dtype=torch.float32, device=self.device)
            batch_pred = self.predict_tensor(batch).detach().cpu().numpy()
            preds.append(batch_pred)

        return np.concatenate(preds, axis=0)


def load_cv_auc(artifact_dir: str | Path, model_name: str) -> float:
    """
    从 artifacts/{model_name}_cv_results.json 中读取平均验证 AUC。
    如果读取失败，则返回 0.5。
    """
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / f"{model_name}_cv_results.json"

    if not path.exists():
        return 0.5

    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        aucs = []
        for item in records:
            if "val_auc" in item:
                aucs.append(float(item["val_auc"]))

        if not aucs:
            return 0.5

        return float(np.mean(aucs))

    except Exception:
        return 0.5

def load_best_params(artifact_dir: str | Path, model_name: str) -> dict:
    """
    读取 artifacts/{model_name}_optuna_best_params.json。
    如果文件不存在，则返回空字典，使用默认模型参数。
    """
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / f"{model_name}_optuna_best_params.json"

    if not path.exists():
        print(f"未找到 {path.name}，将使用默认参数加载 {model_name}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)

        print(f"加载 {model_name} 最优参数: {params}")
        return params

    except Exception as e:
        print(f"读取 {path.name} 失败，原因: {e}，将使用默认参数")
        return {}

def build_auc_weights(
    artifact_dir: str | Path,
    model_names: List[str],
    min_auc: float = 0.5,
) -> Dict[str, float]:
    """
    根据各模型CV AUC构建线性权重。

    逻辑：
        AUC <= 0.5 的模型不加权或权重很低
        AUC 越高，权重越高

    weight_i ∝ max(auc_i - min_auc, 0)
    """
    raw_weights = {}

    for name in model_names:
        auc = load_cv_auc(artifact_dir, name)
        raw_weights[name] = max(auc - min_auc, 0.0)

    total = sum(raw_weights.values())

    if total <= 1e-12:
        return {name: 1.0 / len(model_names) for name in model_names}

    return {name: value / total for name, value in raw_weights.items()}


def build_softmax_auc_weights(
    artifact_dir: str | Path,
    model_names: List[str],
    temperature: float = 0.05,
) -> Dict[str, float]:
    """
    根据 AUC 使用 softmax 构建权重。

    temperature 越小，越偏向 AUC 最高的模型。
    temperature 越大，权重越平均。

    推荐：
        temperature = 0.03 ~ 0.10
    """
    aucs = []

    for name in model_names:
        aucs.append(load_cv_auc(artifact_dir, name))

    aucs = np.asarray(aucs, dtype=float)

    scaled = aucs / max(temperature, 1e-8)
    scaled = scaled - scaled.max()

    exp_values = np.exp(scaled)
    weights = exp_values / exp_values.sum()

    return {
        name: float(weight)
        for name, weight in zip(model_names, weights)
    }


def build_equal_weights(model_names: List[str]) -> Dict[str, float]:
    """
    普通平均权重。
    """
    return {name: 1.0 / len(model_names) for name in model_names}


def load_model_for_ensemble(
    model_name: str,
    model_builder,
    artifact_dir: str | Path,
    input_size: int,
    seq_len: int,
    device: str = "cpu",
    params: Optional[dict] = None,
):
    """
    加载单个模型权重。

    注意：
    加载 .pth 前必须用训练时相同的超参数创建模型。
    因此这里会优先读取：
        artifacts/{model_name}_optuna_best_params.json
    """
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / f"{model_name}_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

    # 如果没有手动传 params，就自动读取 Optuna 最佳参数
    if params is None:
        params = load_best_params(artifact_dir, model_name)

    model = model_builder(
        model_name=model_name,
        input_size=input_size,
        seq_len=seq_len,
        params=params,
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def build_ensemble_from_artifacts(
    model_names: List[str],
    model_builder,
    artifact_dir: str | Path,
    input_size: int,
    seq_len: int,
    device: str = "cpu",
    weighting: str = "softmax_auc",
    model_params_map: Optional[Dict[str, dict]] = None,
) -> WeightedEnsemble:
    """
    从 artifacts 中加载多个模型，并构建 ensemble。

    weighting 可选：
        "equal"       普通平均
        "auc"         按 AUC - 0.5 线性加权
        "softmax_auc" 按 AUC softmax 加权，推荐

    model_params_map:
        如果不同模型训练时用了不同超参数，可以传：
        {
            "lstm": {...},
            "hybrid": {...},
            "cnn": {...},
            "mlp": {...}
        }

        如果不传，则使用 build_model 里的默认参数。
    """
    artifact_dir = Path(artifact_dir)
    model_params_map = model_params_map or {}

    models = {}

    for name in model_names:
        models[name] = load_model_for_ensemble(
            model_name=name,
            model_builder=model_builder,
            artifact_dir=artifact_dir,
            input_size=input_size,
            seq_len=seq_len,
            device=device,
            params=model_params_map.get(name, None),
        )

    if weighting == "equal":
        weights = build_equal_weights(model_names)

    elif weighting == "auc":
        weights = build_auc_weights(
            artifact_dir=artifact_dir,
            model_names=model_names,
            min_auc=0.5,
        )

    elif weighting == "softmax_auc":
        weights = build_softmax_auc_weights(
            artifact_dir=artifact_dir,
            model_names=model_names,
            temperature=0.05,
        )

    else:
        raise ValueError(f"不支持的 weighting: {weighting}")

    print("Ensemble 权重：")
    for name, weight in weights.items():
        auc = load_cv_auc(artifact_dir, name)
        print(f"  {name}: weight={weight:.4f}, cv_auc={auc:.4f}")

    ensemble = WeightedEnsemble(
        models=models,
        weights=weights,
        device=device,
    )

    return ensemble


def save_ensemble_weights(
    weights: Dict[str, float],
    save_path: str | Path,
):
    """
    保存 ensemble 权重。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)