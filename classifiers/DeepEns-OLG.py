
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn


# ================================ IDE运行配置区 =================================
MODE = "predict"  # train 或 predict
VALIDATION_MANIFEST = Path(r"E:\olg\code\validation_manifest.json")
TEST_MANIFEST = Path(r"E:\olg\code\val_test_manifest.json")
VALIDATION_LABELS_TSV: Path | None = Path(r"E:\olg\O-linkedGlycosylation\data\val\val_labels.tsv")
TEST_LABELS_TSV: Path | None = None
LABEL_0_IS_TRUE_NEGATIVE = True

OUTPUT_ROOT = Path(r"E:\olg\O-linkedGlycosylation\global_selected_vote_mlp_models_v2_85")
RUN_NAME = "global_selected_vote_mlp_v4"
CHECKPOINT = Path(
    r"E:\olg\O-linkedGlycosylation\global_selected_vote_mlp_models_v2_85"
    r"\global_selected_vote_mlp_v4\global_selected_vote_mlp.pt"
)
PREDICT_OUTPUT = Path(r"E:\olg\O-linkedGlycosylation\global_selected_vote_mlp_v2_85_val_test\val_test_final_scores.npz")

# 验证集中的模型筛选规则（沿用 v3 的收紧筛选）。
MIN_AP_GAIN = 0.01
MIN_MCC = -1.0
MAX_ABS_PEARSON_CORRELATION = 0.85
MAX_SELECTED_MODELS = 10
MIN_SELECTED_MODELS = 1

# 仅训练最终MLP；筛选和权重始终只由验证集决定。
META_FOLDS = 5
SPLIT_SEED = 42
# V4-5：多 seed 重复分组交叉拟合的种子列表（首个种子用于产出检查点，其余仅报告方差）。
OOF_REPEAT_SEEDS = [42, 1, 7, 2025, 13]
# 主干网络容量（沿用 v3）。
HIDDEN_DIM = 64
SECOND_HIDDEN_DIM = 32
DROPOUT = 0.30
# 早停（沿用 v3）。
EPOCHS = 400
EARLY_STOP_PATIENCE = 25
INNER_VAL_FRACTION = 0.2
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 3e-3
BATCH_SIZE = 128
# Bagging 子模型数。
N_BAG_MODELS = 5
# V4-3：自适应阈值搜索目标与网格（MCC 目标 + 加密到 0.01）。
THRESHOLD_SEARCH_TARGET = "mcc"  # mcc / f1 / weighted
THRESHOLD_WEIGHTED_ALPHA = 0.5    # weighted 目标下 F1 与 MCC 的混合权重
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 3)
EARLY_STOP_THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.05), 3)  # 早停评估用的粗网格，控成本
SCORE_CLIP = 1e-5
CLASSIFICATION_THRESHOLD = 0.5  # 仅作为回退默认值；实际以自适应阈值为准
DEVICE = "auto"  # auto / cuda / cpu
# V4-1：概率校准方法（isotonic / platt / none）。小样本上 platt 更抗过拟合，可按需切换。
CALIBRATION_METHOD = "isotonic"
# V4-2：是否对加权分数/加权总分做 logit 变换后再喂 MLP。
USE_LOGIT_FEATURES = True
# ==============================================================================


ARCHITECTURE = "global_selected_ap_gain_weighted_scores_mlp_v4"
UTILITY_SCRIPT = Path(__file__).with_name("select_weighted_vote_ensemble.py")
AGG_FEATURES = 4  # 加权总分 / 加权最大值 / 加权标准差 / 入选数量比
FEATURE_DIM = MAX_SELECTED_MODELS + AGG_FEATURES


def load_utility():
    if not UTILITY_SCRIPT.is_file():
        raise FileNotFoundError(
            f"缺少依赖脚本：{UTILITY_SCRIPT}。请将select_weighted_vote_ensemble.py与本文件放在同一目录。"
        )
    spec = importlib.util.spec_from_file_location("weighted_vote_utility", UTILITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入：{UTILITY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CLASSIFICATION_THRESHOLD = CLASSIFICATION_THRESHOLD
    module.MIN_AP_GAIN = MIN_AP_GAIN
    module.MIN_MCC = MIN_MCC
    module.MAX_ABS_PEARSON_CORRELATION = MAX_ABS_PEARSON_CORRELATION
    module.MAX_SELECTED_MODELS = MAX_SELECTED_MODELS
    module.MIN_SELECTED_MODELS = MIN_SELECTED_MODELS
    return module


class FinalMLP(nn.Module):
    """两层 MLP，含 BatchNorm+GELU+Dropout（沿用 v3）。"""
    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM,
                 second_hidden: int = SECOND_HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.in_dim = in_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden),
            nn.BatchNorm1d(second_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"MLP输入必须是[B,{self.in_dim}]，实际为{tuple(x.shape)}")
        return self.network(x).squeeze(-1)


def get_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE not in {"cpu", "cuda"}:
        raise ValueError("DEVICE只能为auto、cpu或cuda")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前PyTorch没有可用CUDA")
    return torch.device(DEVICE)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_npz(path: Path, **arrays) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def selected_indices_and_weights(selection: dict) -> tuple[list[int], np.ndarray]:
    """读取按AP排名排列的入选模型，并校验其权重。"""
    indices = [int(item) for item in selection["selected_indices"]]
    weights = np.asarray(selection["weights"], dtype=np.float32)
    if not indices or len(indices) != len(weights) or len(set(indices)) != len(indices):
        raise ValueError("筛选结果的模型索引或权重无效")
    if min(indices) < 0 or max(indices) >= 25 or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("筛选结果索引/权重超出范围")
    if not np.isclose(weights.sum(), 1.0, atol=1e-5):
        raise ValueError("筛选模型权重之和必须为1")
    if len(indices) > MAX_SELECTED_MODELS:
        raise ValueError("入选模型数超过MAX_SELECTED_MODELS，请检查筛选配置")
    return indices, weights


def prob_to_logit(p: np.ndarray, clip: float = SCORE_CLIP) -> np.ndarray:
    """V4-2：概率 -> log-odds，先 clip 防 0/1 处发散。"""
    p = np.clip(np.asarray(p, dtype=np.float64), clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def make_selected_score_features(base_scores: np.ndarray, selection: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 [K个加权分数 | 4个聚合特征]，零填充到固定 FEATURE_DIM。

    沿用 v3 的 stacking 聚合特征；V4-2 在 USE_LOGIT_FEATURES 为真时，
    将“加权分数”与“加权总分”由概率转 log-odds（最大值/std/入选比保持概率口径）。
    """
    if base_scores.ndim != 2 or base_scores.shape[1] != 25:
        raise ValueError("基础模型分数必须为[N,25]")
    indices, weights = selected_indices_and_weights(selection)
    sel = base_scores[:, indices]
    weighted = sel * weights[None, :]

    feats = np.zeros((len(base_scores), FEATURE_DIM), dtype=np.float32)
    weighted_sum = weighted.sum(axis=1)          # 加权总分（归一化投票概率）
    feats[:, :len(indices)] = weighted

    agg = np.zeros((len(base_scores), AGG_FEATURES), dtype=np.float32)
    agg[:, 0] = weighted_sum
    agg[:, 1] = weighted.max(axis=1)             # 加权最大值
    agg[:, 2] = weighted.std(axis=1)             # 加权标准差（模型间分歧度）
    agg[:, 3] = len(indices) / MAX_SELECTED_MODELS  # 入选数量比（信息丰富度）
    feats[:, MAX_SELECTED_MODELS:MAX_SELECTED_MODELS + AGG_FEATURES] = agg

    if USE_LOGIT_FEATURES:
        feats[:, :len(indices)] = prob_to_logit(feats[:, :len(indices)]).astype(np.float32)
        feats[:, MAX_SELECTED_MODELS] = prob_to_logit(weighted_sum).astype(np.float32)

    if feats.shape[1] != FEATURE_DIM or not np.isfinite(feats).all():
        raise RuntimeError("入选模型加权特征构建失败")
    return feats, np.asarray(indices, dtype=np.int64), weights


def standardize(features: np.ndarray, mean: np.ndarray | None = None, scale: np.ndarray | None = None):
    """最终MLP/LR标准化仅在当前训练折拟合；零填充尾部槽位保持为0。"""
    values = np.asarray(features, dtype=np.float64)
    if mean is None:
        mean, scale = values.mean(axis=0), values.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
    output = ((values - mean) / scale).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("MLP特征标准化后出现NaN/Inf")
    return output, np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)


def _mcc(scores: np.ndarray, y_val: np.ndarray, thr: float) -> float:
    pred = (scores >= thr).astype(np.int8)
    if len(set(pred.tolist())) == 1:  # 全同一类 -> MCC无意义
        return -1.0
    try:
        return float(matthews_corrcoef(y_val, pred))
    except Exception:
        return -1.0


def search_threshold(y: np.ndarray, scores: np.ndarray, grid=THRESHOLD_GRID,
                     target=THRESHOLD_SEARCH_TARGET, alpha=THRESHOLD_WEIGHTED_ALPHA) -> float:
    """V4-3：在分数上网格搜索最大化 MCC（默认）的阈值；可选 f1 或 mcc/f1 加权。"""
    best_thr, best_obj = 0.5, -1e9
    for thr in grid:
        pred = (scores >= thr).astype(np.int8)
        if set(pred.tolist()) != {0, 1}:
            continue
        mcc = _mcc(scores, y, float(thr))
        f1 = float(f1_score(y, pred, zero_division=0))
        if target == "mcc":
            obj = mcc
        elif target == "f1":
            obj = f1
        else:  # weighted
            obj = alpha * f1 + (1 - alpha) * mcc
        if obj > best_obj:
            best_obj, best_thr = obj, float(thr)
    return best_thr


def train_mlp(x: np.ndarray, y: np.ndarray, seed: int, device: torch.device,
              split_hint: np.ndarray | None = None) -> FinalMLP:
    """折内再切内部验证集，V4-3：早停按“先搜阈值再算 MCC”评估。"""
    seed_everything(seed)
    model = FinalMLP(in_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                     second_hidden=SECOND_HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    pos_weight = torch.tensor(max(negatives / max(positives, 1), 1.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(seed)

    # 构造内部验证切分
    if len(y) >= 40 and positives >= 2 and negatives >= 2:
        try:
            if split_hint is not None:
                inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                tr_idx, va_idx = next(iter(inner_splitter.split(np.zeros(len(y)), y, split_hint)))
            else:
                from sklearn.model_selection import train_test_split
                tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=INNER_VAL_FRACTION,
                                                  stratify=y, random_state=seed)
            if len(va_idx) >= 5 and set(y[va_idx].tolist()) == {0, 1} and set(y[tr_idx].tolist()) == {0, 1}:
                use_early = True
            else:
                use_early, tr_idx, va_idx = False, np.arange(len(y)), None
        except Exception:
            use_early, tr_idx, va_idx = False, np.arange(len(y)), None
    else:
        use_early, tr_idx, va_idx = False, np.arange(len(y)), None

    steps = max(1, math.ceil(len(tr_idx) / BATCH_SIZE))
    best_obj, best_state, bad_epochs = -2.0, None, 0
    xtr = torch.from_numpy(x[tr_idx]).to(device)
    ytr = torch.from_numpy(y[tr_idx].astype(np.float32)).to(device)

    for epoch in range(EPOCHS):
        model.train()
        for _ in range(steps):
            rows = rng.choice(len(tr_idx), size=min(BATCH_SIZE, len(tr_idx)), replace=len(tr_idx) < BATCH_SIZE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xtr[rows]), ytr[rows])
            if not torch.isfinite(loss):
                raise RuntimeError("最终MLP的BCE损失出现NaN/Inf")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

        if use_early:
            score = predict_single(model, x[va_idx], device)  # sigmoid 概率
            # V4-3：早停内部验证也“先搜阈值再算 MCC”，而非死守 0.5
            cur_thr = search_threshold(y[va_idx], score, grid=EARLY_STOP_THRESHOLD_GRID, target="mcc")
            cur = _mcc(score, y[va_idx], cur_thr)
            if cur > best_obj + 1e-6:
                best_obj, bad_epochs = cur, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= EARLY_STOP_PATIENCE:
                    break

    if use_early and best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval()


@torch.inference_mode()
def predict_single(model: FinalMLP, x: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """单模型概率（sigmoid 概率口径）。

    关键修复（GPU 设备错位 bug）：旧实现写了 `model = model.to(device)` 与 `model = model.cpu()`，
    而 nn.Module 的 .to()/.cpu() 是就地修改（返回同一对象）。早停内部评估会调用本函数，
    于是第一个 epoch 后模型被悄悄挪回 CPU，第二个 epoch 的 `model(xtr[rows])` 出现
    权重在 CPU、输入在 cuda:0 的错位，F.linear 抛
    "Expected all tensors to be on the same device"。纯 CPU 时 .to/.cpu 是空操作故不暴露。
    现改为：以模型当前所在设备为准，只把输入搬到该设备，全程不搬动模型权重，输出统一 .cpu() 回收。
    """
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for start in range(0, len(x), 512):
        batch = torch.from_numpy(np.ascontiguousarray(x[start:start + 512], dtype=np.float32)).to(dev)
        out.append(torch.sigmoid(model(batch)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.inference_mode()
def predict_bag(models: list[FinalMLP], x: np.ndarray, device: torch.device) -> np.ndarray:
    """多子模型概率平均（Bagging）。"""
    if not models:
        raise ValueError("bagging 子模型列表为空")
    probs = np.stack([predict_single(m, x, device) for m in models], axis=0)
    return probs.mean(axis=0).astype(np.float32)


def train_bag(x: np.ndarray, y: np.ndarray, base_seed: int, device: torch.device,
              split_hint: np.ndarray | None = None) -> list[FinalMLP]:
    n = max(1, int(N_BAG_MODELS))
    return [train_mlp(x, y, base_seed + i, device, split_hint=split_hint) for i in range(n)]


def train_lr(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """V4-4：L2 逻辑回归保底基线（class_weight=balanced 处理类别不平衡）。"""
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
    lr.fit(x, y)
    return lr


def lr_proba(lr: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return lr.predict_proba(x)[:, 1].astype(np.float32)


def splits(y: np.ndarray, proteins: np.ndarray, seed: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    if META_FOLDS < 2:
        raise ValueError("META_FOLDS至少为2")
    seed = SPLIT_SEED if seed is None else seed
    splitter = StratifiedGroupKFold(n_splits=META_FOLDS, shuffle=True, random_state=seed)
    result = []
    for train_rows, holdout_rows in splitter.split(np.zeros(len(y)), y, proteins):
        if set(y[train_rows].tolist()) != {0, 1} or set(y[holdout_rows].tolist()) != {0, 1}:
            raise ValueError("每个蛋白质分组折必须同时含0/1；请降低META_FOLDS或调整SPLIT_SEED")
        result.append((train_rows, holdout_rows))
    return result


def fit_calibrator(scores: np.ndarray, y: np.ndarray, method: str = CALIBRATION_METHOD):
    """V4-1：在分组交叉拟合 OOF 分数上拟合校准器，返回可序列化的数值参数。"""
    s = np.clip(np.asarray(scores, dtype=np.float64), SCORE_CLIP, 1.0 - SCORE_CLIP)
    yy = y.astype(np.float64)
    if method == "none":
        return {"method": "none"}
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(s, yy)
        return {"method": "isotonic",
                "x": np.asarray(iso.X_thresholds_, dtype=np.float64).tolist(),
                "y": np.asarray(iso.y_thresholds_, dtype=np.float64).tolist()}
    # platt：对 logit(s) 做一维逻辑回归 -> sigmoid(a*logit+b)
    lo = prob_to_logit(s)
    pl = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    pl.fit(lo.reshape(-1, 1), yy)
    return {"method": "platt", "a": float(pl.coef_[0][0]), "b": float(pl.intercept_[0])}


def apply_calibrator(scores: np.ndarray, cal: dict) -> np.ndarray:
    """V4-1：应用校准器；isotonic 用单调分段线性插值（与 predict 域内一致，域外 clip）。"""
    s = np.asarray(scores, dtype=np.float64)
    if cal is None or cal.get("method") == "none":
        return s.astype(np.float32)
    if cal["method"] == "isotonic":
        xt = np.asarray(cal["x"], dtype=np.float64)
        yt = np.asarray(cal["y"], dtype=np.float64)
        return np.interp(s, xt, yt).astype(np.float32)
    lo = prob_to_logit(s)
    z = cal["a"] * lo + cal["b"]
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def fit_meta_stacking(mlp_cal: np.ndarray, lr_prob: np.ndarray, y: np.ndarray):
    """V4-4：二阶段 stacking，meta 逻辑回归融合 [校准MLP, LR] 两路概率。"""
    feats = np.stack([np.clip(mlp_cal, SCORE_CLIP, 1 - SCORE_CLIP),
                      np.clip(lr_prob, SCORE_CLIP, 1 - SCORE_CLIP)], axis=1)
    meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    meta.fit(feats, y)
    return {"coef": np.asarray(meta.coef_[0], dtype=np.float64).tolist(),
            "intercept": float(meta.intercept_[0])}


def apply_meta_stacking(mlp_cal: np.ndarray, lr_prob: np.ndarray, meta: dict) -> np.ndarray:
    feats = np.stack([np.clip(mlp_cal, SCORE_CLIP, 1 - SCORE_CLIP),
                      np.clip(lr_prob, SCORE_CLIP, 1 - SCORE_CLIP)], axis=1)
    coef = np.asarray(meta["coef"], dtype=np.float64)
    z = feats @ coef + meta["intercept"]
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def strict_oof(base_scores: np.ndarray, y: np.ndarray, proteins: np.ndarray,
               base_order: list[str], utility, device: torch.device, seed: int | None = None):
    """外层验证折的标签不参与其筛选/定权/最终MLP/LR训练（防泄漏）。

    同时产出两路 OOF 概率：MLP-Bag 原始概率、LR 概率。"""
    final_oof_mlp = np.empty(len(y), dtype=np.float32)
    final_oof_lr = np.empty(len(y), dtype=np.float32)
    selected_count_oof = np.empty(len(y), dtype=np.int8)
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    fold_details = []
    for fold, (train_rows, holdout_rows) in enumerate(splits(y, proteins, seed)):
        selection = utility.select_models(base_scores[train_rows], y[train_rows], base_order)
        train_features, selected_indices, _ = make_selected_score_features(base_scores[train_rows], selection)
        holdout_features, _, _ = make_selected_score_features(base_scores[holdout_rows], selection)
        x_train, mean, scale = standardize(train_features)
        x_holdout, _, _ = standardize(holdout_features, mean, scale)
        # MLP 路
        models = train_bag(x_train, y[train_rows], SPLIT_SEED + fold * 100, device,
                           split_hint=proteins[train_rows])
        final_oof_mlp[holdout_rows] = predict_bag(models, x_holdout, device)
        # LR 路（保底基线）
        lr = train_lr(x_train, y[train_rows])
        final_oof_lr[holdout_rows] = lr_proba(lr, x_holdout)
        selected_count_oof[holdout_rows] = len(selected_indices)
        fold_ids[holdout_rows] = fold
        fold_details.append({"fold": fold, "selected_models": selection["selected"]})
    if np.any(fold_ids < 0):
        raise RuntimeError("分组折外预测未覆盖全部验证样本")
    return final_oof_mlp, final_oof_lr, selected_count_oof, fold_ids, fold_details


def state_dict_cpu(model: FinalMLP) -> dict:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def oof_with_calibration_stacking(base_scores, y, proteins, base_order, utility, device, seed):
    """跑一轮分组交叉拟合 -> 校准 -> stacking -> 阈值搜索 -> 诊断。返回该 seed 的全套产物。"""
    oof_mlp, oof_lr, selected_count_oof, fold_ids, fold_details = strict_oof(
        base_scores, y, proteins, base_order, utility, device, seed=seed
    )
    cal = fit_calibrator(oof_mlp, y)
    mlp_cal = apply_calibrator(oof_mlp, cal)
    meta = fit_meta_stacking(mlp_cal, oof_lr, y)
    final_oof = apply_meta_stacking(mlp_cal, oof_lr, meta)
    best_threshold = search_threshold(y, final_oof)
    diagnostic = utility.basic_metrics(y, final_oof, best_threshold)
    return {
        "seed": seed, "oof_mlp": oof_mlp, "oof_lr": oof_lr, "mlp_cal": mlp_cal,
        "cal": cal, "meta": meta, "final_oof": final_oof, "best_threshold": best_threshold,
        "diagnostic": diagnostic, "selected_count_oof": selected_count_oof,
        "fold_ids": fold_ids, "fold_details": fold_details,
    }


def train() -> None:
    if not LABEL_0_IS_TRUE_NEGATIVE:
        raise ValueError("监督式筛选与BCE MLP要求label=0是真实负例；未标注U需采用PU方案")
    if not (0 < CLASSIFICATION_THRESHOLD < 1 and 1 <= MIN_SELECTED_MODELS <= MAX_SELECTED_MODELS <= 25):
        raise ValueError("分类阈值或筛选模型数配置无效")
    if Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {"", ".", ".."}:
        raise ValueError("RUN_NAME必须是单个安全目录名")
    output = OUTPUT_ROOT / RUN_NAME
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    utility = load_utility()
    data = utility.load_scores(VALIDATION_MANIFEST, VALIDATION_LABELS_TSV, require_y=True)
    y = data["y"]
    if set(y.tolist()) != {0, 1}:
        raise ValueError("验证集必须同时包含真实正例和真实负例")
    device = get_device()

    # V4-5：多 seed 重复分组交叉拟合，主 seed 产出检查点，全部 seed 报告方差。
    primary_seed = OOF_REPEAT_SEEDS[0]
    print(f"开始分组交叉拟合（共 {len(OOF_REPEAT_SEEDS)} 个划分种子，主种子={primary_seed}）："
          "每折独立筛选25个模型、定权、bagging训练MLP + L2-LR 保底基线……")
    results: dict[int, dict] = {}
    ap_list, mcc_list = [], []
    for si, sd in enumerate(OOF_REPEAT_SEEDS):
        res = oof_with_calibration_stacking(
            data["scores"], y, data["proteins"], data["base_order"], utility, device, sd)
        results[sd] = res
        ap_list.append(float(res["diagnostic"]["pr_auc"]))
        mcc_list.append(float(res["diagnostic"]["mcc"]))
        print(f"  [seed={sd}] AP={res['diagnostic']['pr_auc']:.4f} | "
              f"MCC={res['diagnostic']['mcc']:.4f} | 自适应阈值={res['best_threshold']:.2f}")
    variance = {
        "seeds": OOF_REPEAT_SEEDS,
        "pr_auc_mean": float(np.mean(ap_list)), "pr_auc_std": float(np.std(ap_list, ddof=0)),
        "mcc_mean": float(np.mean(mcc_list)), "mcc_std": float(np.std(mcc_list, ddof=0)),
    }

    primary = results[primary_seed]
    final_oof = primary["final_oof"]
    best_threshold = primary["best_threshold"]
    cal, meta = primary["cal"], primary["meta"]
    oof_mlp, oof_lr = primary["oof_mlp"], primary["oof_lr"]
    selected_count_oof, fold_ids, fold_details = (
        primary["selected_count_oof"], primary["fold_ids"], primary["fold_details"])
    diagnostic = primary["diagnostic"]
    # 对照：主 seed 下“仅 MLP-Bag 未融合”与“自适应阈值 vs 0.5”。
    diag_mlp_only = utility.basic_metrics(y, oof_mlp, search_threshold(y, oof_mlp))
    diagnostic_at_05 = utility.basic_metrics(y, final_oof, 0.5)

    # 全验证集固定筛选与权重，训练最终 MLP-bag 与 LR 供独立测试使用。
    print("使用完整验证集固定筛选模型和权重，并 bagging 训练最终MLP + L2-LR 供独立测试使用……")
    selection = utility.select_models(data["scores"], y, data["base_order"])
    full_features, full_selected_indices, full_selected_weights = make_selected_score_features(data["scores"], selection)
    x_full, mean, scale = standardize(full_features)
    models = train_bag(x_full, y, SPLIT_SEED + 1000, device, split_hint=data["proteins"])
    lr_full = train_lr(x_full, y)

    output.mkdir(parents=True)
    checkpoint = {
        "architecture": ARCHITECTURE, "base_order": data["base_order"],
        "selected_indices": selection["selected_indices"], "selected_models": selection["selected"],
        "weights": selection["weights"].tolist(),
        "bag_state_dicts": [state_dict_cpu(m) for m in models],
        "feature_dim": FEATURE_DIM,
        "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
        "hidden_dim": HIDDEN_DIM, "second_hidden_dim": SECOND_HIDDEN_DIM, "dropout": DROPOUT,
        "best_threshold": best_threshold,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        # V4 新增：校准器、LR 保底基线、stacking meta（全部为可序列化数值，weights_only 可加载）。
        "calibrator": cal,
        "lr_coef": torch.from_numpy(np.asarray(lr_full.coef_, dtype=np.float32)),
        "lr_intercept": torch.from_numpy(np.asarray(lr_full.intercept_, dtype=np.float32)),
        "meta_stacking": meta,
        "use_logit_features": bool(USE_LOGIT_FEATURES),
        "agg_features": int(AGG_FEATURES),
        "seen_validation_proteins": sorted(set(data["proteins"].tolist())),
        "seen_validation_ids_sha256": hashlib.sha256("\n".join(sorted(data["ids"].tolist())).encode()).hexdigest(),
    }
    checkpoint_path = output / "global_selected_vote_mlp.pt"
    torch.save(checkpoint, checkpoint_path)

    save_json(output / "model_selection_and_oof_diagnostic.json", {
        "architecture": ARCHITECTURE,
        "selection_rules": {"min_ap_gain": MIN_AP_GAIN, "min_mcc": MIN_MCC,
                            "max_abs_pearson_correlation": MAX_ABS_PEARSON_CORRELATION,
                            "max_selected_models": MAX_SELECTED_MODELS,
                            "feature_design": "selected weighted scores(+logit) + 4 stacking aggregates, zero-padded to FEATURE_DIM"},
        "improvements": {
            "feature_augment": "append weighted_sum/max/std and selected_count_ratio (4 aggregates)",
            "logit_features": bool(USE_LOGIT_FEATURES),
            "backbone": f"two-layer MLP {FEATURE_DIM}->{HIDDEN_DIM}->{SECOND_HIDDEN_DIM}->1 with BatchNorm+GELU+Dropout",
            "early_stopping": f"inner group/stratified split, patience={EARLY_STOP_PATIENCE}, max_epochs={EPOCHS}",
            "bagging": f"{N_BAG_MODELS} seeds averaged",
            "probability_calibration": CALIBRATION_METHOD,
            "lr_stacking": "L2 logistic baseline (grouped OOF) fused with calibrated MLP via meta logistic stacking",
            "adaptive_threshold": {"target": THRESHOLD_SEARCH_TARGET, "grid_step": 0.01,
                                   "value": round(best_threshold, 3)},
            "tighter_selection": {"min_ap_gain": MIN_AP_GAIN, "max_abs_pearson": MAX_ABS_PEARSON_CORRELATION},
            "multi_seed_variance": variance,
        },
        "positive_prevalence": selection["positive_prevalence"],
        "selected_models": selection["selected"], "all_models_ranked": selection["ranking"],
        "strict_grouped_oof_diagnostic": diagnostic,
        "strict_grouped_oof_diagnostic_mlp_only": diag_mlp_only,
        "strict_grouped_oof_diagnostic_at_0.5": diagnostic_at_05,
        "fold_selections": fold_details,
    })

    atomic_npz(output / "validation_global_selected_vote_mlp_oof_predictions.npz",
               ids=data["ids"], protein_ids=data["proteins"], y=y, base_scores=data["scores"],
               full_selected_indices=full_selected_indices, full_selected_weights=full_selected_weights,
               grouped_oof_selected_model_count=selected_count_oof,
               oof_mlp_scores=oof_mlp, oof_mlp_calibrated=primary["mlp_cal"], oof_lr_scores=oof_lr,
               final_oof_scores=final_oof,
               predicted_labels=(final_oof >= best_threshold).astype(np.uint8),
               best_threshold=np.asarray(best_threshold), fold_ids=fold_ids,
               metrics_json=np.asarray(json.dumps(diagnostic, ensure_ascii=False, allow_nan=False)))

    print(f"保留模型数={len(selection['selected'])}；自适应阈值({THRESHOLD_SEARCH_TARGET})={best_threshold:.2f}；"
          f"验证集严格折外(stacking)：AP={diagnostic['pr_auc']:.4f} | MCC={diagnostic['mcc']:.4f} | "
          f"AUC={diagnostic['roc_auc']!s} | Accuracy={diagnostic['accuracy']:.4f}")
    print(f"对照仅MLP未融合：AP={diag_mlp_only['pr_auc']:.4f} | MCC={diag_mlp_only['mcc']:.4f}")
    print(f"对照固定阈值0.5：MCC={diagnostic_at_05['mcc']:.4f} | Accuracy={diagnostic_at_05['accuracy']:.4f}")
    print(f"多 seed 方差：AP={variance['pr_auc_mean']:.4f}±{variance['pr_auc_std']:.4f} | "
          f"MCC={variance['mcc_mean']:.4f}±{variance['mcc_std']:.4f}")
    for item in selection["selected"]:
        print(f"  {item['model']}: AP={item['pr_auc']:.4f}, MCC={item['mcc']:.4f}, 权重={item['weight']:.4f}")
    print(f"最终模型检查点：{checkpoint_path}")
    print("验证集已用于模型筛选、定权、校准/stacking与MLP/LR训练；最终独立性能仅在测试集报告。")


def predict() -> None:
    if PREDICT_OUTPUT.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{PREDICT_OUTPUT}")
    metrics_path = PREDICT_OUTPUT.with_name(PREDICT_OUTPUT.stem + "_metrics.json")
    if LABEL_0_IS_TRUE_NEGATIVE and metrics_path.exists():
        raise FileExistsError(f"拒绝覆盖已有指标文件：{metrics_path}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"找不到检查点：{CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("检查点架构与本改进脚本不一致（请用同一份脚本 train/predict；旧 v2/v3 检查点无法加载）")
    if int(checkpoint.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
        raise ValueError("检查点特征维度与当前脚本不一致（AGG_FEATURES/USE_LOGIT 配置已变），请重新训练")
    if int(checkpoint.get("agg_features", AGG_FEATURES)) != AGG_FEATURES:
        raise ValueError("检查点聚合特征数量与当前脚本不一致，请重新训练")
    utility = load_utility()
    data = utility.load_scores(TEST_MANIFEST, TEST_LABELS_TSV, require_y=False)
    if data["base_order"] != checkpoint["base_order"]:
        raise ValueError("测试清单中25个基础模型的顺序与检查点不一致")
    overlap = set(data["proteins"].tolist()) & set(checkpoint["seen_validation_proteins"])
    if overlap:
        raise ValueError(f"测试集与筛选验证集重叠{len(overlap)}个蛋白质，拒绝作为独立测试")
    selection = {"selected_indices": checkpoint["selected_indices"], "weights": checkpoint["weights"]}
    features, selected_indices, selected_weights = make_selected_score_features(data["scores"], selection)
    mean, scale = checkpoint["mean"].cpu().numpy(), checkpoint["scale"].cpu().numpy()
    x, _, _ = standardize(features, mean, scale)

    in_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
    device = get_device()
    bag_dicts = checkpoint.get("bag_state_dicts") or [checkpoint["state_dict"]]
    models = []
    for sd in bag_dicts:
        m = FinalMLP(in_dim=in_dim, hidden_dim=int(checkpoint["hidden_dim"]),
                     second_hidden=int(checkpoint.get("second_hidden_dim", SECOND_HIDDEN_DIM)),
                     dropout=float(checkpoint["dropout"]))
        m.load_state_dict(sd)
        models.append(m.to(device).eval())
    raw_mlp = predict_bag(models, x, device)
    # V4-1：应用训练阶段保存的校准器。
    mlp_cal = apply_calibrator(raw_mlp, checkpoint.get("calibrator"))
    # V4-4：用全验证集训练的 LR 产概率。
    lr_coef = checkpoint["lr_coef"].cpu().numpy()
    lr_int = checkpoint["lr_intercept"].cpu().numpy()
    z_lr = x @ lr_coef.T + lr_int
    lr_prob = (1.0 / (1.0 + np.exp(-z_lr[:, 0]))).astype(np.float32)
    meta = checkpoint.get("meta_stacking")
    if meta:
        scores = apply_meta_stacking(mlp_cal, lr_prob, meta)
    else:
        scores = mlp_cal

    threshold = float(checkpoint.get("best_threshold", checkpoint["classification_threshold"]))
    arrays = {"ids": data["ids"], "protein_ids": data["proteins"], "base_scores": data["scores"],
              "selected_indices": selected_indices, "selected_weights": selected_weights,
              "raw_mlp_scores": raw_mlp, "mlp_calibrated": mlp_cal, "lr_scores": lr_prob,
              "scores": scores, "predicted_labels": (scores >= threshold).astype(np.uint8),
              "classification_threshold": np.asarray(threshold), "checkpoint": np.asarray(str(CHECKPOINT))}
    report = None
    if data["y"] is not None:
        arrays["y"] = data["y"]
        if LABEL_0_IS_TRUE_NEGATIVE:
            report = utility.basic_metrics(data["y"], scores, threshold)
            arrays["metrics_json"] = np.asarray(json.dumps(report, ensure_ascii=False, allow_nan=False))
            arrays["confusion_matrix"] = np.asarray(report["confusion_matrix"], dtype=np.int64)
    atomic_npz(PREDICT_OUTPUT, **arrays)
    print(f"已导出测试集最终预测：{PREDICT_OUTPUT}")
    if report is not None:
        save_json(metrics_path, report)
        print(f"测试集：AP={report['pr_auc']:.4f} | MCC={report['mcc']:.4f} | AUC={report['roc_auc']!s} | "
              f"Accuracy={report['accuracy']:.4f} | Recall={report['recall']:.4f} | Precision={report['precision']:.4f}")


if __name__ == "__main__":
    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise ValueError("MODE 只能为 train 或 predict")