from pathlib import Path
import hashlib
import json
import math
import random
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ================= IDE 配置区 =================
MODE = "predict"                       # train 或 predict
TRAIN_NPZ = Path(r"E:\olg\O-linkedGlycosylation\data\balanced_train_5\encoded_aaindex\train_balanced_05_aaindex.npz")
OUTPUT_ROOT = Path(r"E:\olg\O-linkedGlycosylation\data\ccg_aa_v2_models")
RUN_NAME = "dataset_05_v2"              # 五个数据集分别改成 dataset_01 ... dataset_05
LOSS_MODE = "bce"                   # 默认真实负例；未标注数据才使用 nnpu
PI_U = None                         # 仅 nnpu 必填；BCE 不使用
CLASSIFICATION_THRESHOLD = 0.5      # 固定准确率判定阈值，不在每轮自动寻找阈值
SEED = 42
SPLIT_SEED = 42                     # 所有编码/五个子集应保持相同
INTERNAL_VAL_FRACTION = 0.2
EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-5
BATCH_SIZE = 32                     # 每批一半标签1、一半标签0，有放回抽样
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
DEVICE = "auto"                    # auto / cpu / cuda
# 仅 MODE=predict 时使用；不会训练或调参，可用于原验证集的融合特征。
CHECKPOINT = Path(r"E:\olg\O-linkedGlycosylation\data\ccg_aa_v2_models\dataset_01_v2\best_model.pt")
PREDICT_NPZ = Path(r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_aaindex.npz")
PREDICT_OUTPUT = Path(r"E:\olg\O-linkedGlycosylation\data\ccg_aa_val_test_v2\val_test_01.npz")
PREDICT_LABELS_ARE_TRUE_PN = True    # 预测文件的 y: 1=真实正例、0=真实负例；若0是U，必须设False
# =============================================


class LocalResidualBlock(nn.Module):
    """并行深度卷积提取局部模式，逐点卷积混合通道，FFN增加非线性。"""
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.branch3 = nn.Conv1d(channels, channels, 3, padding=dilation,
                                 dilation=dilation, groups=channels)
        self.branch5 = nn.Conv1d(channels, channels, 5, padding=2*dilation,
                                 dilation=dilation, groups=channels)
        self.mix = nn.Conv1d(2*channels, channels, 1)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 2*channels), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(2*channels, channels))
        self.drop = nn.Dropout(0.1)

    def forward(self, h):
        z = self.norm1(h).transpose(1, 2)
        local = self.mix(torch.cat((F.gelu(self.branch3(z)),
                                   F.gelu(self.branch5(z))), dim=1)).transpose(1, 2)
        h = h + self.drop(local)
        return h + self.drop(self.ffn(self.norm2(h)))


class CCGAA(nn.Module):
    """V2：残差扩张卷积+方向性统计池化；保留类名以复用训练入口。
    不是旧模型权重兼容实现，必须重新训练。无padding，未知残基仍参与建模。
    """
    def __init__(self, input_dim, length=21, center=10, channels=32):
        super().__init__()
        if not 0 < center < length-1:
            raise ValueError("中心位置必须同时具有左右上下文")
        self.length, self.center, self.input_dim = length, center, input_dim
        self.projection = nn.Linear(input_dim, channels)
        self.unknown = nn.Parameter(torch.zeros(channels))
        self.position = nn.Parameter(torch.zeros(1, length, channels))
        self.input_drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(LocalResidualBlock(channels, 1),
                                    LocalResidualBlock(channels, 2))
        self.head = nn.Sequential(nn.LayerNorm(5*channels),
                                  nn.Linear(5*channels, 64), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(64, 32),nn.ReLU(),nn.Linear(32, 1))

    def forward(self, x, known_mask):
        if (x.ndim != 3 or x.shape[1:] != (self.length, self.input_dim)
                or known_mask.shape != x.shape[:2]):
            raise ValueError("输入应为[B,L,D]且与模型配置一致，掩码应为[B,L]")
        h = F.gelu(self.projection(x))
        h = h + (~known_mask.bool()).unsqueeze(-1).to(h.dtype)*self.unknown
        h = self.input_drop(h + self.position)
        t = self.blocks(h)
        c = self.center
        left, right = t[:, :c], t[:, c+1:]
        # 均值保留整体背景，最大值保留各通道较强的局部响应，中心始终独立保留。
        pooled = torch.cat((left.mean(1), left.amax(1), t[:, c],
                            right.mean(1), right.amax(1)), dim=-1)
        return self.head(pooled).squeeze(-1)


def load_data(path, require_labels=True):
    with np.load(path, allow_pickle=False) as f:
        required = {"X", "ids", "known_residue_mask", "index_ids", "lookup_table",
                    "standardized", "center_position_1_based"}
        if missing := required - set(f.files):
            raise ValueError(f"{path} 缺少字段：{sorted(missing)}；请使用更新后的 AAindex 编码程序")
        data = {k: f[k].copy() for k in f.files}
    x = np.asarray(data["X"], dtype=np.float32)
    ids = data["ids"].astype(str)
    mask = data["known_residue_mask"]
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("X 必须为非空、有限数值的 [N,L,D]")
    if ids.shape != (len(x),) or len(set(ids.tolist())) != len(ids) or any(not s for s in ids):
        raise ValueError("样本 ID 为空、重复或数量错误")
    if mask.shape != x.shape[:2] or not np.isin(mask, [0, 1]).all():
        raise ValueError("known_residue_mask 的形状或值无效")
    indices = data["index_ids"].astype(str)
    if indices.shape != (x.shape[2],) or len(set(indices.tolist())) != len(indices):
        raise ValueError("AAindex 指标数量错误或重复")
    lookup = np.asarray(data["lookup_table"], dtype="<f4")
    if lookup.shape != (21, x.shape[2]) or not np.isfinite(lookup).all():
        raise ValueError("lookup_table 应为 [21,D] 的有限数值矩阵")
    center = int(data["center_position_1_based"].item()) - 1
    if not 0 < center < x.shape[1] - 1:
        raise ValueError("无效的中心位置")
    proteins = data.get("protein_ids", np.asarray([re.sub(r"_\d+$", "", s) for s in ids])).astype(str)
    if proteins.shape != ids.shape or any(not p for p in proteins):
        raise ValueError("缺少有效蛋白质 ID；请在编码时保存 protein_ids")
    if "protein_ids" not in data and any(not re.search(r"_\d+$", s) for s in ids):
        raise ValueError("无法从 ID 提取蛋白质名称，要求 protein_siteNumber 格式")
    y = data.get("y")
    if y is None and require_labels:
        raise ValueError("训练文件缺少 y；编码时需要提供标签表")
    if y is not None:
        if y.shape != ids.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("y 必须为逐样本的 0/1 标签")
        y = y.astype(np.int64)
    schema = {"index_ids": indices.tolist(), "length": x.shape[1], "input_dim": x.shape[2],
              "center": center, "standardized": bool(data["standardized"].item()),
              "lookup_sha256": hashlib.sha256(lookup.tobytes()).hexdigest()}
    return {"X": x, "ids": ids, "mask": mask.astype(bool), "proteins": proteins,
            "y": y, "schema": schema}


def split_groups(proteins, y, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("内部验证比例必须在 (0,1)")
    # 按蛋白质 ID 稳定哈希：跨子集不会将同一蛋白质同时放入训练与早停集。
    def is_val(protein):
        digest = hashlib.sha256(f"{seed}|{protein}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < fraction
    val_mask = np.asarray([is_val(p) for p in proteins])
    train, val = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    for name, rows in (("内部训练", train), ("内部验证", val)):
        if len(rows) == 0 or set(y[rows].tolist()) != {0, 1}:
            raise ValueError(f"{name}必须同时有标签 1/0；数据太少或分组不适合。请统一调整所有模型的划分")
    return train, val


def risk_parts(logits, y, prior):
    positive, unlabeled = logits[y == 1], logits[y == 0]
    if not len(positive) or not len(unlabeled):
        raise ValueError("nnPU 风险计算必须同时包含 P 与 U")
    positive_risk = prior * torch.sigmoid(-positive).mean()
    negative_risk = torch.sigmoid(unlabeled).mean() - prior * torch.sigmoid(positive).mean()
    return positive_risk, negative_risk


def training_loss(logits, y, mode, prior):
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, y.float())
    rp, rn = risk_parts(logits, y, prior)
    # 原 nnPU 的 beta=0, gamma=1 纠正梯度；不是简单 clamp 后令负风险梯度消失。
    return -rn if rn.detach().item() < 0 else rp + rn


def validation_risk(logits, y, mode, prior):
    # 在整个早停集上计算 P/U 条件均值，不能平均各推理批次的 PU 风险。
    z, labels = torch.as_tensor(logits), torch.as_tensor(y)
    if mode == "bce":
        return float(F.binary_cross_entropy_with_logits(z, labels.float()))
    rp, rn = risk_parts(z, labels, prior)
    return float(rp + rn.clamp_min(0))


def validation_accuracy(logits, y, threshold=0.5):
    """真实正负例准确率；sigmoid(logit)>=threshold 判正，等号包含在正例中。"""
    if not 0 < threshold < 1:
        raise ValueError("分类阈值必须在 (0,1)")
    logits, y = np.asarray(logits), np.asarray(y)
    if logits.ndim != 1 or logits.shape != y.shape or not len(y):
        raise ValueError("准确率输入必须是等长非空的一维数组")
    if not np.isfinite(logits).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("准确率输入包含无效分数或标签")
    cutoff = math.log(threshold / (1 - threshold))
    return float(np.mean((logits >= cutoff) == y))


def get_device():
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查环境或设 DEVICE='cpu'")
    return device


@torch.inference_mode()
def predict_logits(model, data, rows, device, batch_size=128):
    model.eval()
    results = []
    for start in range(0, len(rows), batch_size):
        ix = rows[start:start+batch_size]
        x = torch.from_numpy(data["X"][ix]).to(device)
        mask = torch.from_numpy(data["mask"][ix]).to(device)
        results.append(model(x, mask).cpu().numpy())
    result = np.concatenate(results)
    if not np.isfinite(result).all():
        raise RuntimeError("预测出现 NaN/Inf")
    return result


def ranking_metrics(logits, y):
    """直接使用 logit 排序，避免 sigmoid 在极端分数处饱和而引入假并列。
    ROC-AUC：逐个同分数组处理，等价于并列分数使用平均秩。
    AP：每个不同阈值处 sum(召回率增量 * 精确率)，不是 PR 曲线梯形积分。
    本函数由 classification_metrics 在完成输入校验后调用。
    """
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    # 降序排序；相同分数作为一个组，避免 AP 依赖组内样本顺序。
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(labels) - 1]
    tp = np.cumsum(sorted_y, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    delta_tp = np.diff(np.r_[0, tp])
    delta_fp = np.diff(np.r_[0, fp])
    auc = None
    if positives and negatives:
        previous_tp = np.r_[0, tp[:-1]]
        # 积分 TPR dFPR：并列分数组内的正负对计半分。
        auc = float(np.sum(delta_fp * (previous_tp + 0.5 * delta_tp))
                    / (float(positives) * negatives))
    ap = None
    if positives:
        precision = tp / (ends + 1)
        ap = float(np.sum((delta_tp / positives) * precision))
    return auc, ap


def classification_metrics(logits, y, threshold=0.5):
    """真实 PN 二分类评价。单类/零分母指标保留为 None，不伪造为 0。"""
    accuracy = validation_accuracy(logits, y, threshold)
    y = np.asarray(y, dtype=np.int64)
    predicted = (np.asarray(logits) >= math.log(threshold / (1 - threshold))).astype(np.int64)
    # 行=真实类别，列=预测类别，固定顺序 [0,1]。
    matrix = np.bincount(2 * y + predicted, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    n = len(y)
    positives, negatives = tp + fn, tn + fp
    undefined = {}

    def divide(name, numerator, denominator, reason):
        if denominator == 0:
            undefined[name] = reason
            return None
        return float(numerator / denominator)

    precision = divide("precision", tp, tp + fp, "没有预测为正例的样本")
    recall = divide("recall", tp, positives, "没有真实正例")
    specificity = divide("specificity", tn, negatives, "没有真实负例")
    npv = divide("negative_predictive_value", tn, tn + fn, "没有预测为负例的样本")
    f1 = divide("f1", 2 * tp, 2 * tp + fp + fn, "真实标签和预测标签均没有正例")
    balanced = None
    if recall is not None and specificity is not None:
        balanced = float((recall + specificity) / 2)
    else:
        undefined["balanced_accuracy"] = "需要同时存在真实正例和真实负例"
    # Python int 避免 NumPy int64 连乘溢出。
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = divide("mcc", tp * tn - fp * fn, mcc_denominator,
                 "真实标签或预测标签只有一个类别，MCC 分母为零")
    roc_auc, ap = ranking_metrics(logits, y)
    if roc_auc is None:
        undefined["roc_auc"] = "ROC-AUC 需要同时存在真实正例和真实负例"
    if ap is None:
        undefined["average_precision"] = "没有真实正例，AP 未定义"
    warnings = []
    if not positives or not negatives:
        warnings.append("测试集只有一个真实类别，不能完整评价二分类能力。")
    if positives and not negatives:
        warnings.append("全为正例时 AP=1，不代表模型具有区分正负例的能力。")
    return {
        "accuracy": accuracy, "roc_auc": roc_auc, "average_precision": ap,
        "mcc": mcc, "balanced_accuracy": balanced,
        "precision": precision, "recall": recall, "specificity": specificity,
        "negative_predictive_value": npv, "f1": f1,
        "correct_count": int(np.trace(matrix)), "sample_count": n,
        "positive_count": positives, "negative_count": negatives,
        "predicted_positive_count": tp + fp, "predicted_negative_count": tn + fn,
        "positive_prevalence": float(positives / n),
        "majority_class_accuracy_baseline": float(max(positives, negatives) / n),
        "ap_prevalence_reference": float(positives / n),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(), "label_order": [0, 1],
        "matrix_axes": "rows=true, columns=predicted",
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "ranking_score": "logits (before sigmoid; higher means more positive)",
        "average_precision_definition": "sum(delta_recall * precision); NOT trapezoidal PR-AUC",
        "mcc_zero_denominator_policy": "undefined -> JSON null / NPZ NaN",
        "undefined_metrics": undefined, "warnings": warnings,
    }


def print_classification_metrics(metrics):
    print(f"评价样本={metrics['sample_count']}；正例={metrics['positive_count']}；"
          f"负例={metrics['negative_count']}；正例比例={metrics['positive_prevalence']:.2%}")
    for key, title in (
        ("accuracy", "准确率 Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "PR 曲线指标 AP（Average Precision，非梯形积分）"),
        ("mcc", "MCC"),
        ("balanced_accuracy", "平衡准确率 Balanced Accuracy"),
        ("precision", "精确率 Precision"),
        ("recall", "召回率 Recall / Sensitivity"),
        ("specificity", "特异度 Specificity"),
        ("f1", "F1"),
        ("negative_predictive_value", "阴性预测值 NPV"),
        ("majority_class_accuracy_baseline", "全部预测为多数类的 Accuracy 基线"),
        ("ap_prevalence_reference", "AP 的正例比例参考值"),
    ):
        value = metrics[key]
        if value is None:
            print(f"{title}=未定义（{metrics['undefined_metrics'].get(key, '数据条件不满足')}）")
        else:
            print(f"{title}={value:.6f}")
    for warning in metrics["warnings"]:
        print(f"注意：{warning}")


def export_predictions(path, data, rows, logits, checkpoint, role, threshold=None, metrics=None):
    payload = {"ids": data["ids"][rows], "protein_ids": data["proteins"][rows],
               "logits": logits, "scores": torch.sigmoid(torch.from_numpy(logits)).numpy(),
               "checkpoint": np.asarray(str(checkpoint)), "data_role": np.asarray(role),
               "score_semantics": np.asarray("sigmoid score; not necessarily calibrated probability")}
    if data["y"] is not None:
        payload["y"] = data["y"][rows]
    if threshold is not None:
        payload["predicted_labels"] = (logits >= math.log(threshold / (1-threshold))).astype(np.uint8)
        payload["classification_threshold"] = np.asarray(threshold)
    if metrics is not None:
        payload["confusion_matrix"] = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        payload["accuracy"] = np.asarray(metrics["accuracy"])
        payload["label_order"] = np.asarray([0, 1])
        payload["matrix_axes"] = np.asarray(metrics["matrix_axes"])
        # NPZ 既保存可直接读取的标量指标，也保存完整 JSON（含未定义原因）。
        for key in (
            "roc_auc", "average_precision", "mcc", "balanced_accuracy",
            "precision", "recall", "specificity", "negative_predictive_value", "f1",
            "positive_prevalence", "majority_class_accuracy_baseline",
            "ap_prevalence_reference",
        ):
            payload[key] = np.asarray(np.nan if metrics[key] is None else metrics[key],
                                      dtype=np.float64)
        payload["metrics_json"] = np.asarray(
            json.dumps(metrics, ensure_ascii=False, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)


def train():
    if LOSS_MODE not in {"nnpu", "bce"}:
        raise ValueError("LOSS_MODE 只能是 nnpu 或 bce")
    if LOSS_MODE == "nnpu" and (PI_U is None or not 0 < PI_U < 1):
        raise ValueError("请先设置 PI_U：从训练数据估计的 U 池阳性比例；不能因为 P/U 平衡就填 0.5")
    if not 0 < CLASSIFICATION_THRESHOLD < 1:
        raise ValueError("CLASSIFICATION_THRESHOLD 必须在 (0,1)")
    if EPOCHS < 1 or PATIENCE < 1 or BATCH_SIZE < 2 or BATCH_SIZE % 2:
        raise ValueError("EPOCHS/PATIENCE 需为正数，BATCH_SIZE 需为正偶数且至少为 2")
    if Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {"", ".", ".."}:
        raise ValueError("RUN_NAME 只能为单个目录名称")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(SEED)
    data = load_data(TRAIN_NPZ)
    tr, va = split_groups(data["proteins"], data["y"], INTERNAL_VAL_FRACTION, SPLIT_SEED)
    device = get_device()
    schema = data["schema"]
    kwargs = {k: schema[k] for k in ("input_dim", "length", "center")}
    model = CCGAA(**kwargs).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output = OUTPUT_ROOT / f"{RUN_NAME}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best_model.pt"
    config = {"train_npz": str(TRAIN_NPZ.resolve()), "loss_mode": LOSS_MODE,
              "pi_u": PI_U if LOSS_MODE == "nnpu" else None,
              "classification_threshold": CLASSIFICATION_THRESHOLD,
              "selection_metric": "val_loss" if LOSS_MODE == "bce" else "val_risk",
              "seed": SEED, "split_seed": SPLIT_SEED, "internal_val_fraction": INTERNAL_VAL_FRACTION,
              "epochs": EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
              "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
              "architecture": "ccg_aa_v2_residual_pool", "parameters": parameter_count, "device": str(device), "torch_version": str(torch.__version__),
              "label_semantics": "1=P,0=U" if LOSS_MODE == "nnpu" else "1=P,0=verified N",
              "schema": schema, "sampling": "equal label-1/label-0 batches; uniform replacement within each pool",
              "note": "internal validation predictions are NOT independent stacking training features"}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output / "split_ids.npz", train_ids=data["ids"][tr], val_ids=data["ids"][va],
                        train_proteins=data["proteins"][tr], val_proteins=data["proteins"][va])
    print(f"设备={device}；参数={parameter_count:,}；内部训练={len(tr)}；内部验证={len(va)}")
    if LOSS_MODE == "bce":
        print(f"BCE 二分类：1=正例，0=真实负例；准确率阈值={CLASSIFICATION_THRESHOLD}；按验证损失选择模型。")
    print(f"输出目录：{output}")
    pools = [tr[data["y"][tr] == label] for label in (1, 0)]
    steps = math.ceil(len(tr) / BATCH_SIZE)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        objectives = []
        for _ in range(steps):
            rows = np.concatenate([rng.choice(pool, BATCH_SIZE // 2, replace=True) for pool in pools])
            rng.shuffle(rows)
            x = torch.from_numpy(data["X"][rows]).to(device)
            mask = torch.from_numpy(data["mask"][rows]).to(device)
            y = torch.from_numpy(data["y"][rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            objective = training_loss(model(x, mask), y, LOSS_MODE, PI_U)
            if not torch.isfinite(objective):
                raise RuntimeError("训练损失出现 NaN/Inf")
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            objectives.append(objective.item())
        logits = predict_logits(model, data, va, device)
        risk = validation_risk(logits, data["y"][va], LOSS_MODE, PI_U)
        accuracy = (validation_accuracy(logits, data["y"][va], CLASSIFICATION_THRESHOLD)
                    if LOSS_MODE == "bce" else None)
        history.append({"epoch": epoch, "train_update_objective": float(np.mean(objectives)),
                        "val_risk": risk, "val_loss": risk if LOSS_MODE == "bce" else None,
                        "val_accuracy": accuracy})
        # update_objective 在 nnPU 纠正分支并非完整风险，不能与 val_risk 直接对比。
        improved = risk < best - MIN_DELTA
        if improved:
            best, stale = risk, 0
            torch.save({"state_dict": model.state_dict(), "model_kwargs": kwargs,
                        "schema": schema, "config": config, "epoch": epoch, "val_risk": risk,
                        "val_accuracy": accuracy,
                        "seen_proteins": sorted(set(data["proteins"].tolist())),
                        "seen_ids": data["ids"].tolist()}, checkpoint)
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        progress = (f"训练损失={np.mean(objectives):.6f} | 内部验证损失={risk:.6f}"
                    f" | 内部验证准确率={accuracy:.2%}" if LOSS_MODE == "bce" else
                    f"更新目标={np.mean(objectives):.6f} | 内部验证风险={risk:.6f}")
        print(f"Epoch {epoch:03d} | {progress}"
              + (" | 保存最佳权重" if improved else ""))
        if stale >= PATIENCE:
            print("达到早停条件")
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    logits = predict_logits(model, data, va, device)
    export_predictions(output / "internal_val_predictions.npz", data, va, logits, checkpoint,
                       "early_stopping_only_NOT_independent_meta_training")
    print(f"完成。最佳 epoch={saved['epoch']}，模型：{checkpoint}")
    if LOSS_MODE == "bce":
        print(f"最佳验证损失模型对应的内部验证准确率={saved['val_accuracy']:.2%}")
    print("内部验证参与了模型选择，不能冒充独立 stacking 特征；原验证集请通过 predict 模式处理。")


def predict():
    if PREDICT_OUTPUT.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{PREDICT_OUTPUT}")
    if PREDICT_OUTPUT.resolve() in {PREDICT_NPZ.resolve(), CHECKPOINT.resolve()}:
        raise ValueError("输出不能覆盖输入")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if saved.get("config", {}).get("architecture") != "ccg_aa_v2_residual_pool":
        raise ValueError("V2只能加载V2训练的权重；旧版权重请用原脚本预测，不能直接迁移")
    data = load_data(PREDICT_NPZ, require_labels=False)
    threshold = float(saved.get("config", {}).get("classification_threshold", CLASSIFICATION_THRESHOLD))
    if not 0 < threshold < 1:
        raise ValueError("模型保存的分类阈值必须在 (0,1)")
    evaluate = data["y"] is not None and PREDICT_LABELS_ARE_TRUE_PN
    metrics_path = PREDICT_OUTPUT.with_name(PREDICT_OUTPUT.stem + "_metrics.json")
    if evaluate:
        if metrics_path.exists():
            raise FileExistsError(f"拒绝覆盖已有评估结果：{metrics_path}")
        if metrics_path.resolve() in {PREDICT_NPZ.resolve(), CHECKPOINT.resolve()}:
            raise ValueError("评估结果路径不能覆盖输入")
    if data["schema"] != saved["schema"]:
        raise ValueError("编码不匹配：请核对指标顺序、查找表、标准化、长度及中心位置")
    overlap = set(data["proteins"].tolist()) & set(saved["seen_proteins"])
    if overlap:
        raise ValueError(f"预测集与基层训练/早停数据重叠 {len(overlap)} 个蛋白质，拒绝生成独立融合特征")
    device = get_device()
    model = CCGAA(**saved["model_kwargs"]).to(device)
    model.load_state_dict(saved["state_dict"])
    rows = np.arange(len(data["ids"]))
    logits = predict_logits(model, data, rows, device)
    metrics = classification_metrics(logits, data["y"], threshold) if evaluate else None
    export_predictions(PREDICT_OUTPUT, data, rows, logits, CHECKPOINT,
                       "held_out_from_base_training_and_selection", threshold, metrics)
    print(f"已导出 {len(rows)} 个样本的分数：{PREDICT_OUTPUT}")
    if metrics is not None:
        report = dict(metrics, input_npz=str(PREDICT_NPZ.resolve()),
                      checkpoint=str(CHECKPOINT.resolve()), label_semantics="1=P,0=verified N")
        with metrics_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"预测分类阈值={threshold:g}（优先使用模型保存的阈值）")
        print("混淆矩阵：行=真实类别，列=预测类别，0=负例，1=正例")
        print("                  预测0       预测1")
        print(f"真实0（负例） {metrics['tn']:10d}  {metrics['fp']:10d}")
        print(f"真实1（正例） {metrics['fn']:10d}  {metrics['tp']:10d}")
        print(f"预测准确率（Accuracy）={metrics['accuracy']:.2%} "
              f"({metrics['correct_count']}/{metrics['sample_count']})")
        print_classification_metrics(metrics)
        print(f"评估结果：{metrics_path}")
    else:
        reason = "预测文件没有 y 标签" if data["y"] is None else "标签未声明为真实正负例"
        print(f"{reason}，跳过所有监督评价指标；仍导出分数和预测类别。")
    print("与其他编码模型融合前必须按 ids 对齐；本程序未训练外层 MLP。")


def main():
    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise ValueError("MODE 只能为 train 或 predict")


if __name__ == "__main__":
    main()
