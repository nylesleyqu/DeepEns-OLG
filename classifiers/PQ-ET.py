#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ================================ 运行配置区 =================================
MODE = "predict"  # train 或 predict
TRAIN_NPZ = Path(
    r"E:\olg\O-linkedGlycosylation\data\balanced_train_5\encoded_188d"
    r"\train_balanced_05_188d.npz"
)
OUTPUT_ROOT = Path(r"E:\olg\O-linkedGlycosylation\data\ctd_188d_extratrees_models")
RUN_NAME = "dataset_05_ctd_et"

# BCE/树模型监督训练都要求 0 是真实负例；如果0是未标注样本，请使用 nnPU 模型。
CONFIRM_LABEL_0_IS_TRUE_NEGATIVE = True
USE_PLACEHOLDER_QUALITY_FEATURES = True
INTERNAL_VAL_FRACTION = 0.20
SPLIT_SEED = 42
RANDOM_SEED = 42

# ExtraTrees 参数。188D 特征高度相关，min_samples_leaf=2 可降低过拟合风险。
N_ESTIMATORS = 600
MAX_FEATURES = 0.50
MIN_SAMPLES_LEAF = 2
MAX_DEPTH = None
CLASS_WEIGHT = "balanced"
N_JOBS = -1
CLASSIFICATION_THRESHOLD = 0.5

# MODE=predict 时使用。
MODEL_FILE = Path(
    r"E:\olg\O-linkedGlycosylation\data\ctd_188d_extratrees_models"
    r"\dataset_01_ctd_et\model.joblib"
)
PREDICT_NPZ = Path(r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_188d.npz")
PREDICT_OUTPUT = Path(
    r"E:\olg\O-linkedGlycosylation\data\ctd_188d_extratrees_val_test\val_test_01.npz"
)
PREDICT_LABELS_ARE_TRUE_PN = True
# =============================================================================


ARCHITECTURE = "ExtraTreesClassifier_188D_v1"


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def protein_id(sample_id: str) -> str:
    protein, separator, position = sample_id.rpartition("_")
    if not protein or not separator or not position.isdigit():
        raise ValueError(f"无法从 {sample_id!r} 解析 protein_id；应使用 蛋白名_位点 格式")
    return protein


def load_data(path: Path, require_labels: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到编码文件：{path}")
    with np.load(path, allow_pickle=False) as source:
        required = {"X", "ids"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"{path} 缺少字段：{sorted(missing)}")
        raw_x = np.asarray(source["X"], dtype=np.float32)
        ids = source["ids"].astype(str)
        names = (source["feature_names"].astype(str).tolist()
                 if "feature_names" in source.files else [f"ctd_feature_{i + 1:03d}" for i in range(188)])
        y = np.asarray(source["y"], dtype=np.int64) if "y" in source.files else None
        proteins = source["protein_ids"].astype(str) if "protein_ids" in source.files else None
        center = int(source["center_position_1_based"].item()) if "center_position_1_based" in source.files else 11
        fraction = np.asarray(source["known_residue_fraction"], dtype=np.float32) \
            if "known_residue_fraction" in source.files else None
        count = np.asarray(source["known_residue_count"], dtype=np.float32) \
            if "known_residue_count" in source.files else None
        transitions = np.asarray(source["valid_adjacent_transition_count"], dtype=np.float32) \
            if "valid_adjacent_transition_count" in source.files else None
        known_mask = np.asarray(source["known_residue_mask"], dtype=bool) \
            if "known_residue_mask" in source.files else None

    if raw_x.ndim != 2 or raw_x.shape[1] != 188 or not len(raw_x) or not np.isfinite(raw_x).all():
        raise ValueError(f"{path}: X 必须为有限的非空 [N,188]，实际={raw_x.shape}")
    if ids.shape != (len(raw_x),) or not all(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{path}: ids 与 X 不匹配、为空或重复")
    if len(names) != 188 or len(set(names)) != 188:
        raise ValueError(f"{path}: feature_names 必须含 188 个不重复特征名")
    if y is None and require_labels:
        raise ValueError(f"{path}: 训练数据缺少 y")
    if y is not None and (y.shape != ids.shape or not np.isin(y, [0, 1]).all()):
        raise ValueError(f"{path}: y 必须是与 ids 对齐的0/1标签")
    if proteins is None:
        proteins = np.asarray([protein_id(item) for item in ids])
    if proteins.shape != ids.shape or not all(proteins) or len(set(proteins.tolist())) < 2:
        raise ValueError(f"{path}: protein_ids 无效")

    quality, quality_names = [], []
    if USE_PLACEHOLDER_QUALITY_FEATURES:
        fraction = fraction if fraction is not None else np.ones(len(ids), dtype=np.float32)
        count = count if count is not None else np.full(len(ids), 21, dtype=np.float32)
        transitions = transitions if transitions is not None else np.full(len(ids), 20, dtype=np.float32)
        if any(value.shape != ids.shape or not np.isfinite(value).all()
               for value in (fraction, count, transitions)):
            raise ValueError(f"{path}: 占位符质量特征无效")
        if known_mask is not None:
            if known_mask.ndim != 2 or known_mask.shape[0] != len(ids):
                raise ValueError(f"{path}: known_residue_mask 形状无效")
            center_known = known_mask[:, min(center - 1, known_mask.shape[1] - 1)].astype(np.float32)
        else:
            center_known = np.ones(len(ids), dtype=np.float32)
        quality = [fraction, count / 21.0, transitions / 20.0, center_known]
        quality_names = ["known_residue_fraction", "known_residue_count_div_21",
                         "valid_adjacent_transition_count_div_20", "center_residue_known"]
    x = np.concatenate((raw_x, np.column_stack(quality).astype(np.float32)), axis=1) if quality else raw_x
    schema = {"encoding": "188D_CTD", "feature_names": names + quality_names,
              "ctd_dimension": 188, "input_dimension": int(x.shape[1]),
              "center_position_1_based": center,
              "quality_features_enabled": bool(USE_PLACEHOLDER_QUALITY_FEATURES)}
    return {"X": x, "ids": ids, "proteins": proteins, "y": y, "schema": schema}


def split_by_protein(proteins: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < INTERNAL_VAL_FRACTION < 1:
        raise ValueError("INTERNAL_VAL_FRACTION 必须在 (0,1)")

    def is_val(protein: str) -> bool:
        digest = hashlib.sha256(f"{SPLIT_SEED}|{protein}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < INTERNAL_VAL_FRACTION

    val_mask = np.asarray([is_val(item) for item in proteins])
    train_rows, val_rows = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    for title, rows in (("内部训练集", train_rows), ("内部验证集", val_rows)):
        if not len(rows) or set(y[rows].tolist()) != {0, 1}:
            raise ValueError(f"{title}必须同时含标签0和1；请调整SPLIT_SEED或验证比例")
    return train_rows, val_rows


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    if not 0 < threshold < 1:
        raise ValueError("分类阈值必须在 (0,1)")
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(item) for item in matrix.ravel())
    has_both_classes = len(np.unique(y)) == 2
    has_both_predictions = len(np.unique(predicted)) == 2
    return {
        "threshold": float(threshold), "sample_count": int(len(y)),
        "positive_count": int(y.sum()), "negative_count": int(len(y) - y.sum()),
        "accuracy": float(accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, scores)) if has_both_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else None,
        "mcc": float(matthews_corrcoef(y, predicted)) if has_both_classes and has_both_predictions else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if has_both_classes else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(), "matrix_axes": "rows=true, columns=predicted",
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def scores_from_model(model: ExtraTreesClassifier, x: np.ndarray) -> np.ndarray:
    classes = model.classes_.tolist()
    if classes != [0, 1]:
        raise RuntimeError(f"模型类别异常：{classes}")
    scores = model.predict_proba(x)[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("预测概率出现NaN/Inf")
    return scores.astype(np.float32)


def export_importance(path: Path, names: list[str], values: np.ndarray) -> None:
    order = np.argsort(-values)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write("rank\tfeature_name\timportance\n")
        for rank, index in enumerate(order, start=1):
            handle.write(f"{rank}\t{names[index]}\t{float(values[index]):.10g}\n")


def build_model() -> ExtraTreesClassifier:
    if N_ESTIMATORS < 1 or MIN_SAMPLES_LEAF < 1 or not 0 < MAX_FEATURES <= 1:
        raise ValueError("N_ESTIMATORS、MIN_SAMPLES_LEAF或MAX_FEATURES配置无效")
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS, criterion="gini", max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
        class_weight=CLASS_WEIGHT, random_state=RANDOM_SEED, n_jobs=N_JOBS,
    )


def train() -> None:
    if not CONFIRM_LABEL_0_IS_TRUE_NEGATIVE:
        raise ValueError("此监督树模型要求标签0是真实负例；若为U，请使用nnPU。")
    if Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {"", ".", ".."}:
        raise ValueError("RUN_NAME必须是单个安全目录名")
    data = load_data(TRAIN_NPZ, require_labels=True)
    train_rows, val_rows = split_by_protein(data["proteins"], data["y"])
    output = OUTPUT_ROOT / RUN_NAME
    output.mkdir(parents=True, exist_ok=False)
    model = build_model()
    print(f"训练样本={len(train_rows)}；内部验证={len(val_rows)}；特征维度={data['X'].shape[1]}；树数={N_ESTIMATORS}")
    model.fit(data["X"][train_rows], data["y"][train_rows])
    scores = scores_from_model(model, data["X"][val_rows])
    report = metrics(data["y"][val_rows], scores, CLASSIFICATION_THRESHOLD)

    model_file = output / "model.joblib"
    joblib.dump({"model": model, "schema": data["schema"],
                 "seen_proteins": sorted(set(data["proteins"].tolist())),
                 "seen_ids": data["ids"].tolist(), "architecture": ARCHITECTURE,
                 "classification_threshold": CLASSIFICATION_THRESHOLD}, model_file)
    config = {"architecture": ARCHITECTURE, "train_npz": str(TRAIN_NPZ.resolve()),
              "label_semantics": "1=verified positive, 0=verified negative",
              "internal_val_fraction": INTERNAL_VAL_FRACTION, "split_seed": SPLIT_SEED,
              "random_seed": RANDOM_SEED, "n_estimators": N_ESTIMATORS,
              "max_features": MAX_FEATURES, "min_samples_leaf": MIN_SAMPLES_LEAF,
              "max_depth": MAX_DEPTH, "class_weight": CLASS_WEIGHT, "n_jobs": N_JOBS,
              "classification_threshold": CLASSIFICATION_THRESHOLD, "schema": data["schema"],
              "selection_note": "fixed configuration; internal validation is evaluation only"}
    save_json(output / "config.json", config)
    atomic_npz(output / "split_ids.npz", train_ids=data["ids"][train_rows], val_ids=data["ids"][val_rows],
               train_protein_ids=data["proteins"][train_rows], val_protein_ids=data["proteins"][val_rows])
    atomic_npz(output / "internal_val_predictions.npz", ids=data["ids"][val_rows],
               protein_ids=data["proteins"][val_rows], y=data["y"][val_rows], scores=scores,
               predicted_labels=(scores >= CLASSIFICATION_THRESHOLD).astype(np.uint8),
               classification_threshold=np.asarray(CLASSIFICATION_THRESHOLD),
               confusion_matrix=np.asarray(report["confusion_matrix"], dtype=np.int64),
               metrics_json=np.asarray(json.dumps(report, ensure_ascii=False, allow_nan=False)))
    save_json(output / "internal_val_metrics.json", report)
    export_importance(output / "feature_importance.tsv", data["schema"]["feature_names"], model.feature_importances_)
    print(f"内部验证：Accuracy={report['accuracy']:.4f} | AUC={report['roc_auc']!s} | "
          f"MCC={report['mcc']!s} | AP={report['average_precision']!s} | "
          f"Balanced Accuracy={report['balanced_accuracy']!s}")
    print(f"模型：{model_file}")


def predict() -> None:
    if PREDICT_OUTPUT.exists():
        raise FileExistsError(f"拒绝覆盖已有预测：{PREDICT_OUTPUT}")
    metrics_path = PREDICT_OUTPUT.with_name(PREDICT_OUTPUT.stem + "_metrics.json")
    if PREDICT_LABELS_ARE_TRUE_PN and metrics_path.exists():
        raise FileExistsError(f"拒绝覆盖已有指标文件：{metrics_path}")
    saved = joblib.load(MODEL_FILE)
    if saved.get("architecture") != ARCHITECTURE:
        raise ValueError("不是本 188D ExtraTrees 脚本产生的模型文件")
    data = load_data(PREDICT_NPZ, require_labels=False)
    if data["schema"] != saved["schema"]:
        raise ValueError("预测编码 schema 不一致：请核对188D列顺序及占位符质量特征设置")
    overlap = set(data["proteins"].tolist()) & set(saved["seen_proteins"])
    if overlap:
        raise ValueError(f"预测集与基层训练/验证集重叠 {len(overlap)} 个蛋白质，拒绝生成独立预测")
    scores = scores_from_model(saved["model"], data["X"])
    threshold = float(saved["classification_threshold"])
    arrays = {"ids": data["ids"], "protein_ids": data["proteins"], "scores": scores,
              "predicted_labels": (scores >= threshold).astype(np.uint8),
              "classification_threshold": np.asarray(threshold), "model_file": np.asarray(str(MODEL_FILE))}
    report = None
    if data["y"] is not None:
        arrays["y"] = data["y"]
        if PREDICT_LABELS_ARE_TRUE_PN:
            report = metrics(data["y"], scores, threshold)
            arrays["confusion_matrix"] = np.asarray(report["confusion_matrix"], dtype=np.int64)
            arrays["metrics_json"] = np.asarray(json.dumps(report, ensure_ascii=False, allow_nan=False))
    atomic_npz(PREDICT_OUTPUT, **arrays)
    print(f"已导出 {len(scores)} 个分数：{PREDICT_OUTPUT}")
    if report is not None:
        save_json(metrics_path, report)
        print(f"Accuracy={report['accuracy']!s} | AUC={report['roc_auc']!s} | MCC={report['mcc']!s} | "
              f"AP={report['average_precision']!s} | Balanced Accuracy={report['balanced_accuracy']!s}")


def main() -> None:
    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise ValueError("MODE只能是 train 或 predict")


if __name__ == "__main__":
    main()
