from pathlib import Path
import hashlib
import json
import math
import random
import re
from datetime import datetime

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ================= IDE 配置区 =================
MODE = "predict"
TRAIN_H5 = Path(r"E:\olg\O-linkedGlycosylation\data\balanced_train_5\encoded_prott5\train_balanced_05_prott5.h5")
OUTPUT_ROOT = Path(r"E:\olg\O-linkedGlycosylation\data\prott5_conv_attention_models")
RUN_NAME = "dataset_05_conv_attention"
LOSS_MODE = "bce"
PI_U = None
CLASSIFICATION_THRESHOLD = 0.5
EXPECTED_LENGTH = 21
CENTER_POSITION = 11
EXPECTED_INPUT_DIM = 1024
CHANNELS = 32
ATTENTION_HEADS = 2
USE_CONV = True
USE_ATTENTION = True
DROPOUT = 0.2
CONFIRMED_TRAIN_NEGATIVES = True  # 用户当前假设0是真实负例；若未证实，请勿使用BCE
SEED = 42
SPLIT_SEED = 42
INTERNAL_VAL_FRACTION = 0.2
EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-5
BATCH_SIZE = 32
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0       # 先关闭以便解释损失；可独立消融
DEVICE = "auto"

CHECKPOINT = Path(r"E:\olg\O-linkedGlycosylation\data\prott5_conv_attention_models\dataset_01_conv_attention\best_model.pt")
PREDICT_H5 = Path(r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_prott5.h5")
PREDICT_OUTPUT = Path(r"E:\olg\O-linkedGlycosylation\data\prott5_conv_attention_val_test\val_test_01.npz")
PREDICT_LABELS_ARE_TRUE_PN = True
# =============================================

ARCHITECTURE = "prott5_conv_center_attention_v1"


class ConvCenterAttentionProtT5(nn.Module):
    """局部多尺度卷积+单个中心查询的多头交叉注意力。
    不划分左右区域，不使用原型。注意力权重不是生物学因果解释。
    """
    def __init__(self, input_dim=1024, length=21, center=10, channels=64,
                 heads=2, dropout=0.2, use_conv=True, use_attention=True):
        super().__init__()
        if not 0 < center < length-1:
            raise ValueError("中心必须同时具有左右上下文")
        if channels < 1 or heads < 1 or channels % heads:
            raise ValueError("channels必须为正且可被heads整除")
        if not 0 <= dropout < 1:
            raise ValueError("dropout必须在[0,1)")
        self.input_dim, self.length, self.center = input_dim, length, center
        self.channels, self.heads = channels, heads
        self.use_conv, self.use_attention = use_conv, use_attention
        self.project = nn.Sequential(nn.Linear(input_dim, channels), nn.GELU(),
                                     nn.LayerNorm(channels))
        self.input_drop = nn.Dropout(0.2)
        if use_conv:
            self.conv3 = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
            self.conv5 = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
            self.mix = nn.Conv1d(2*channels, channels, 1)
            self.local_drop = nn.Dropout(0.1)
            self.local_norm = nn.LayerNorm(channels)
        indices = [i for i in range(length) if i != center]
        self.register_buffer("context_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        if use_attention:
            self.q = nn.Linear(channels, channels, bias=False)
            self.k = nn.Linear(channels, channels, bias=False)
            self.v = nn.Linear(channels, channels, bias=False)
            self.out = nn.Linear(channels, channels, bias=False)
            # 与context_indices一一对应；固定窗口时为中心相对位置偏置。
            self.relative_bias = nn.Parameter(torch.zeros(heads, length-1))
            self.attn_drop = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.Linear(3*channels, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, return_attention=False):
        if x.ndim != 3 or x.shape[1:] != (self.length,self.input_dim):
            raise ValueError("输入必须是与配置一致的[B,L,D]逐残基嵌入")
        h = self.input_drop(self.project(x.to(dtype=self.project[0].weight.dtype)))
        if self.use_conv:
            z = h.transpose(1,2)
            local = self.mix(torch.cat((F.gelu(self.conv3(z)), F.gelu(self.conv5(z))),dim=1))
            h = self.local_norm(h+self.local_drop(local.transpose(1,2)))
        center = h[:,self.center]
        context = h.index_select(1,self.context_indices)
        batch, positions, _ = context.shape
        if self.use_attention:
            d = self.channels//self.heads
            query = self.q(center).reshape(batch,self.heads,d)
            key = self.k(context).reshape(batch,positions,self.heads,d).transpose(1,2)
            value = self.v(context).reshape(batch,positions,self.heads,d).transpose(1,2)
            scores = (query.unsqueeze(2)*key).sum(-1)/math.sqrt(d)
            weights = torch.softmax(scores+self.relative_bias.unsqueeze(0),dim=-1)
            # dropout仅用于训练时汇总，返回的原始权重每个头仍加和为1。
            pooled = (self.attn_drop(weights).unsqueeze(-1)*value).sum(2).reshape(batch,self.channels)
            pooled = self.out(pooled)
        else:
            pooled = context.mean(1)
            weights = context.new_full((batch,self.heads,positions),1.0/positions)
        combined = torch.cat((center,pooled,center*pooled),dim=-1)
        logits = self.head(combined).squeeze(-1)
        if return_attention:
            return logits, weights
        return logits


class H5Residues:
    """每批打开只读文件，排序/去重索引后恢复请求顺序，支持PU重复采样。
    不常驻全部嵌入，不持有跨进程HDF5句柄。文件在训练期间不应被修改。
    """
    def __init__(self, path, shape):
        self.path, self.shape = Path(path), tuple(shape)

    def __getitem__(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0:
            raise ValueError("批次索引必须为非空一维数组")
        if (rows < 0).any() or (rows >= self.shape[0]).any():
            raise IndexError("样本索引超出范围")
        unique, inverse = np.unique(rows, return_inverse=True)
        with h5py.File(self.path, "r") as f:
            block = np.asarray(f["residue_embeddings"][unique], dtype=np.float32)[inverse]
        if not np.isfinite(block).all():
            raise ValueError("逐残基嵌入存在NaN/Inf，停止训练或预测")
        return block


def load_data(path, require_labels=True):
    """与此前prott5_encode_fragments.py的HDF5字段一致。"""
    path = Path(path)
    with h5py.File(path, "r") as f:
        missing = {"residue_embeddings", "ids", "sequences"} - set(f.keys())
        if missing:
            raise ValueError(f"{path} 缺少{sorted(missing)}；需要逐残基编码，不能使用mean/center向量替代")
        required_attrs = {"complete", "completed_samples", "sample_count", "fragment_length",
                          "hidden_size", "center_position_1_based", "model_name_or_path"}
        if missing := required_attrs - set(f.attrs):
            raise ValueError(f"缺少编码元数据：{sorted(missing)}")
        shape = f["residue_embeddings"].shape
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("residue_embeddings必须为非空[N,L,D]")
        n, length, dimension = shape
        if not bool(f.attrs["complete"]) or int(f.attrs["completed_samples"]) != n or int(f.attrs["sample_count"]) != n:
            raise ValueError("编码未完成或完成数量不一致，拒绝读取可能未写入的特征")
        if length != int(f.attrs["fragment_length"]) or dimension != int(f.attrs["hidden_size"]):
            raise ValueError("特征形状和编码元数据不一致")
        center_1 = int(f.attrs["center_position_1_based"])
        if (length, dimension, center_1) != (EXPECTED_LENGTH, EXPECTED_INPUT_DIM, CENTER_POSITION):
            raise ValueError("编码长度、维度或中心位置与配置不一致")
        ids = np.asarray(f["ids"].asstr()[:], dtype=str)
        seqs = np.asarray(f["sequences"].asstr()[:], dtype=str)
        if ids.shape != (n,) or len(set(ids.tolist())) != n or any(not v for v in ids):
            raise ValueError("样本ID为空、重复或数量错误")
        if seqs.shape != (n,) or any(len(v) != length for v in seqs):
            raise ValueError("序列数量或长度与特征不一致")
        if "protein_ids" in f:
            proteins = np.asarray(f["protein_ids"].asstr()[:], dtype=str)
        else:
            if any(not re.fullmatch(r".+_\d+", identifier) for identifier in ids):
                raise ValueError("缺少protein_ids，样本ID须为蛋白质名_数字位点格式")
            proteins = np.asarray([v.rsplit("_",1)[0] for v in ids])
        if proteins.shape != (n,) or any(not v for v in proteins):
            raise ValueError("蛋白质ID无效")
        y = f["labels"][:] if "labels" in f else None
        if y is None and require_labels:
            raise ValueError("训练编码缺少labels；编码时需提供标签表")
        if y is not None:
            if y.shape != (n,) or not np.isin(y,[0,1]).all():
                raise ValueError("labels须为逐样本0/1标签")
            y = y.astype(np.int64)
        def attr_text(key, default=""):
            value = f.attrs.get(key, default)
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        schema = {"encoding":"ProtT5_residue", "input_dim": dimension, "length":length,
                  "center":center_1-1, "model_name_or_path":attr_text("model_name_or_path"),
                  "non_standard_residue_policy":attr_text("non_standard_residue_policy","map_to_X")}
        source_semantics = attr_text("label_semantics","not_recorded")
    return {"X":H5Residues(path,shape), "ids":ids, "proteins":proteins, "y":y,
            "schema":schema, "source_label_semantics":source_semantics}


def split_groups(proteins, y, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("内部验证比例必须在 (0,1)")

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


def training_loss(logits, y, mode, prior, label_smoothing=0.0):
    if mode == "bce":
        target = y.float()
        if label_smoothing > 0:
            # 把 {0,1} 平滑到 {eps/2, 1-eps/2}
            target = target * (1 - label_smoothing) + 0.5 * label_smoothing
        return F.binary_cross_entropy_with_logits(logits, target)
    rp, rn = risk_parts(logits, y, prior)
    return -rn if rn.detach().item() < 0 else rp + rn


def validation_risk(logits, y, mode, prior):
    z, labels = torch.as_tensor(logits), torch.as_tensor(y)
    if mode == "bce":
        return float(F.binary_cross_entropy_with_logits(z, labels.float()))
    rp, rn = risk_parts(z, labels, prior)
    return float(rp + rn.clamp_min(0))


def validation_accuracy(logits, y, threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError("分类阈值必须在 (0,1)")
    logits, y = np.asarray(logits), np.asarray(y)
    if logits.ndim != 1 or logits.shape != y.shape or not len(y):
        raise ValueError("准确率输入必须是等长非空的一维数组")
    if not np.isfinite(logits).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("准确率输入包含无效分数或标签")
    cutoff = math.log(threshold / (1 - threshold))
    return float(np.mean((logits >= cutoff) == y))


def compute_auc(scores, labels):
    """基于平均秩的 AUC（Mann-Whitney U），等价于 sklearn.metrics.roc_auc_score。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_mcc(logits, y, threshold=0.5):
    cutoff = math.log(threshold / (1 - threshold))
    pred = (np.asarray(logits) >= cutoff).astype(np.int64)
    y = np.asarray(y, dtype=np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


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
        ix = rows[start:start + batch_size]
        x = torch.from_numpy(data["X"][ix]).to(device)
        results.append(model(x).cpu().numpy())
    result = np.concatenate(results)
    if not np.isfinite(result).all():
        raise RuntimeError("预测出现 NaN/Inf")
    return result


def classification_metrics(logits, y, threshold=0.5):
    accuracy = validation_accuracy(logits, y, threshold)
    predicted = (np.asarray(logits) >= math.log(threshold / (1 - threshold))).astype(np.int64)
    matrix = np.bincount(2 * np.asarray(y, dtype=np.int64) + predicted,
                         minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    balanced = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {"precision": precision, "recall": recall, "specificity": specificity,
            "f1": f1, "balanced_accuracy": balanced,
            "accuracy": accuracy, "correct_count": int(np.trace(matrix)),
            "sample_count": int(matrix.sum()), "threshold": float(threshold),
            "auc": compute_auc(logits, y), "mcc": compute_mcc(logits, y, threshold),
            "confusion_matrix": matrix.tolist(), "label_order": [0, 1],
            "matrix_axes": "rows=true, columns=predicted",
            "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])}


def export_predictions(path, data, rows, logits, checkpoint, role, threshold=None, metrics=None):
    payload = {"ids": data["ids"][rows], "protein_ids": data["proteins"][rows],
               "logits": logits, "scores": torch.sigmoid(torch.from_numpy(logits)).numpy(),
               "checkpoint": np.asarray(str(checkpoint)), "data_role": np.asarray(role),
               "score_semantics": np.asarray("sigmoid score; not necessarily calibrated probability")}
    if data["y"] is not None:
        payload["y"] = data["y"][rows]
    if threshold is not None:
        payload["predicted_labels"] = (logits >= math.log(threshold / (1 - threshold))).astype(np.uint8)
        payload["classification_threshold"] = np.asarray(threshold)
    if metrics is not None:
        payload["confusion_matrix"] = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        payload["accuracy"] = np.asarray(metrics["accuracy"])
        payload["auc"] = np.asarray(np.nan if metrics["auc"] is None else metrics["auc"])
        payload["mcc"] = np.asarray(metrics["mcc"])
        payload["label_order"] = np.asarray([0, 1])
        payload["matrix_axes"] = np.asarray(metrics["matrix_axes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)


def train():
    if not 0 <= LABEL_SMOOTHING < 1:
        raise ValueError("LABEL_SMOOTHING必须在[0,1)")
    if not 0 <= DROPOUT < 1 or LEARNING_RATE <= 0 or WEIGHT_DECAY < 0:
        raise ValueError("Dropout/学习率/权重衰减配置无效")
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
    if LOSS_MODE == "bce" and not CONFIRMED_TRAIN_NEGATIVES:
        raise ValueError("BCE要求确认0为真实负例；未标注候选位点应使用PU方案")
    data = load_data(TRAIN_H5)
    print(f"源文件标签说明：{data['source_label_semantics']}")
    if LOSS_MODE == "bce":
        print("当前按用户确认的真实P/N标签训练，不会改写源文件标签说明。")
    tr, va = split_groups(data["proteins"], data["y"], INTERNAL_VAL_FRACTION, SPLIT_SEED)
    device = get_device()
    schema = data["schema"]
    kwargs = {k: schema[k] for k in ("input_dim", "length", "center")}
    kwargs.update(channels=CHANNELS, heads=ATTENTION_HEADS, dropout=DROPOUT,
                  use_conv=USE_CONV, use_attention=USE_ATTENTION)
    model = ConvCenterAttentionProtT5(**kwargs).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, threshold=MIN_DELTA,
        threshold_mode="abs", min_lr=LEARNING_RATE * 0.02)

    output = OUTPUT_ROOT / f"{RUN_NAME}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best_model.pt"
    config = {"train_h5": str(TRAIN_H5.resolve()), "source_label_semantics": data["source_label_semantics"],
              "confirmed_train_negatives": CONFIRMED_TRAIN_NEGATIVES, "loss_mode": LOSS_MODE,
              "pi_u": PI_U if LOSS_MODE == "nnpu" else None,
              "classification_threshold": CLASSIFICATION_THRESHOLD,
              "selection_metric": "val_loss" if LOSS_MODE == "bce" else "val_risk",
              "label_smoothing": LABEL_SMOOTHING,
              "scheduler": "ReduceLROnPlateau(factor=0.5, patience=4)",
              "seed": SEED, "split_seed": SPLIT_SEED, "internal_val_fraction": INTERNAL_VAL_FRACTION,
              "epochs": EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
              "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
              "architecture": ARCHITECTURE, "parameters": parameter_count,
              "device": str(device), "torch_version": str(torch.__version__),
              "label_semantics": "1=P,0=U" if LOSS_MODE == "nnpu" else "1=P,0=verified N",
              "schema": schema,
              "sampling": "shuffle once per epoch without replacement" if LOSS_MODE == "bce" else "balanced PU with replacement",
              "note": "internal validation predictions are NOT independent stacking training features"}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output / "split_ids.npz", train_ids=data["ids"][tr], val_ids=data["ids"][va],
                        train_proteins=data["proteins"][tr], val_proteins=data["proteins"][va])
    print(f"设备={device}；参数={parameter_count:,}；内部训练={len(tr)}；内部验证={len(va)}")
    if LOSS_MODE == "bce":
        print(f"BCE 二分类：1=正例，0=真实负例；准确率阈值={CLASSIFICATION_THRESHOLD}；按验证损失选择模型。")
    for name, rows in (("内部训练", tr), ("内部验证", va)):
        count = int(data["y"][rows].sum())
        print(f"{name}：标签1={count}，标签0={len(rows)-count}，多数类基线={max(count,len(rows)-count)/len(rows):.2%}")
    print(f"输出目录：{output}")
    pools = [tr[data["y"][tr] == label] for label in (1, 0)]
    steps = math.ceil(len(tr) / BATCH_SIZE)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        objectives, batch_sizes = [], []
        if LOSS_MODE == "bce":
            epoch_rows = rng.permutation(tr)
            batches = [epoch_rows[i:i+BATCH_SIZE] for i in range(0, len(tr), BATCH_SIZE)]
        else:
            batches = [np.concatenate([rng.choice(pool, BATCH_SIZE // 2, replace=True)
                                       for pool in pools]) for _ in range(steps)]
        for rows in batches:
            rng.shuffle(rows)
            x = torch.from_numpy(data["X"][rows]).to(device)
            y = torch.from_numpy(data["y"][rows]).to(device)
            optimizer.zero_grad(set_to_none=True)
            objective = training_loss(model(x), y, LOSS_MODE, PI_U,
                                      label_smoothing=LABEL_SMOOTHING)
            if not torch.isfinite(objective):
                raise RuntimeError("训练损失出现 NaN/Inf")
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            objectives.append(objective.item())
            batch_sizes.append(len(rows))

        logits = predict_logits(model, data, va, device)
        risk = validation_risk(logits, data["y"][va], LOSS_MODE, PI_U)
        y_va = data["y"][va]
        metrics = classification_metrics(logits, y_va, CLASSIFICATION_THRESHOLD) if LOSS_MODE == "bce" else None
        accuracy = metrics["accuracy"] if metrics else None
        auc = metrics["auc"] if metrics else None
        mcc = metrics["mcc"] if metrics else None
        train_logits = predict_logits(model, data, tr, device)
        train_risk = validation_risk(train_logits, data["y"][tr], LOSS_MODE, PI_U)
        train_accuracy = validation_accuracy(train_logits, data["y"][tr], CLASSIFICATION_THRESHOLD) if metrics else None
        update_mean = float(np.average(objectives, weights=batch_sizes))
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_update_objective": update_mean, "train_eval_loss": train_risk,
                        "train_eval_accuracy": train_accuracy, "val_metrics": metrics,
                        "val_risk": risk, "val_loss": risk if LOSS_MODE == "bce" else None,
                        "val_accuracy": accuracy, "val_auc": auc, "val_mcc": mcc,
                        "lr": current_lr})
        improved = risk < best - MIN_DELTA
        if improved:
            best, stale = risk, 0
            torch.save({"state_dict": model.state_dict(), "model_kwargs": kwargs,
                        "schema": schema, "config": config, "epoch": epoch, "val_risk": risk,
                        "val_accuracy": accuracy, "val_auc": auc, "val_mcc": mcc,
                        "seen_proteins": sorted(set(data["proteins"].tolist())),
                        "seen_ids": data["ids"].tolist()}, checkpoint)
        else:
            stale += 1
        scheduler.step(risk)

        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        progress = (f"训练损失={update_mean:.6f} | 训练评估损失={train_risk:.6f} | 内部验证损失={risk:.6f}"
                    f" | 训练准确率={train_accuracy:.2%} | 验证准确率={accuracy:.2%} | AUC={auc:.4f} | MCC={mcc:.4f}"
                    f" | 平衡准确率={metrics['balanced_accuracy']:.2%} | 召回率={metrics['recall']:.2%} | lr={current_lr:.2e}"
                    if LOSS_MODE == "bce" else
                    f"更新目标={np.mean(objectives):.6f} | 内部验证风险={risk:.6f}"
                    f" | lr={current_lr:.2e}")
        print(f"Epoch {epoch:03d} | {progress}" + (" | 保存最佳权重" if improved else ""))
        if stale >= PATIENCE:
            print("达到早停条件")
            break

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    logits = predict_logits(model, data, va, device)
    final_metrics = classification_metrics(logits, data["y"][va], CLASSIFICATION_THRESHOLD) if LOSS_MODE == "bce" else None
    export_predictions(output / "internal_val_predictions.npz", data, va, logits, checkpoint,
                       "early_stopping_only_NOT_independent_meta_training", CLASSIFICATION_THRESHOLD, final_metrics)
    if final_metrics:
        (output / "internal_val_metrics.json").write_text(
            json.dumps(final_metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"完成。最佳 epoch={saved['epoch']}，模型：{checkpoint}")
    if LOSS_MODE == "bce":
        print(f"最佳模型内部验证：loss={saved['val_risk']:.6f}，"
              f"acc={saved['val_accuracy']:.2%}，AUC={saved['val_auc']:.4f}，MCC={saved['val_mcc']:.4f}")
    print("内部验证参与了模型选择，不能冒充独立 stacking 特征；原验证集请通过 predict 模式处理。")


def predict():
    if PREDICT_OUTPUT.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{PREDICT_OUTPUT}")
    if PREDICT_OUTPUT.resolve() in {PREDICT_H5.resolve(), CHECKPOINT.resolve()}:
        raise ValueError("输出不能覆盖输入")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if saved.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError(f"请加载本脚本({ARCHITECTURE})生成的权重")
    data = load_data(PREDICT_H5, require_labels=False)
    threshold = float(saved.get("config", {}).get("classification_threshold", CLASSIFICATION_THRESHOLD))
    if not 0 < threshold < 1:
        raise ValueError("模型保存的分类阈值必须在 (0,1)")
    evaluate = data["y"] is not None and PREDICT_LABELS_ARE_TRUE_PN
    metrics_path = PREDICT_OUTPUT.with_name(PREDICT_OUTPUT.stem + "_metrics.json")
    if evaluate:
        if metrics_path.exists():
            raise FileExistsError(f"拒绝覆盖已有评估结果：{metrics_path}")
        if metrics_path.resolve() in {PREDICT_H5.resolve(), CHECKPOINT.resolve()}:
            raise ValueError("评估结果路径不能覆盖输入")
    if data["schema"] != saved["schema"]:
        raise ValueError("编码不匹配：请核对ProtT5模型来源、特征维度、长度及中心位置")
    overlap = set(data["proteins"].tolist()) & set(saved["seen_proteins"])
    if overlap:
        raise ValueError(f"预测集与基层训练/早停数据重叠 {len(overlap)} 个蛋白质，拒绝生成独立融合特征")
    device = get_device()
    model = ConvCenterAttentionProtT5(**saved["model_kwargs"]).to(device)
    model.load_state_dict(saved["state_dict"])
    rows = np.arange(len(data["ids"]))
    logits = predict_logits(model, data, rows, device)
    metrics = classification_metrics(logits, data["y"], threshold) if evaluate else None
    export_predictions(PREDICT_OUTPUT, data, rows, logits, CHECKPOINT,
                       "held_out_from_base_training_and_selection", threshold, metrics)
    print(f"已导出 {len(rows)} 个样本的分数：{PREDICT_OUTPUT}")
    if metrics is not None:
        report = dict(metrics, input_h5=str(PREDICT_H5.resolve()),
                      checkpoint=str(CHECKPOINT.resolve()), label_semantics="1=P,0=verified N")
        with metrics_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"预测分类阈值={threshold:g}")
        print("混淆矩阵：行=真实类别，列=预测类别，0=负例，1=正例")
        print("                  预测0       预测1")
        print(f"真实0（负例） {metrics['tn']:10d}  {metrics['fp']:10d}")
        print(f"真实1（正例） {metrics['fn']:10d}  {metrics['tp']:10d}")
        print(f"准确率={metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['sample_count']})"
              f" | AUC={metrics['auc']} | MCC={metrics['mcc']:.4f}")
        print(f"评估结果：{metrics_path}")
    else:
        reason = "预测文件没有 y 标签" if data["y"] is None else "标签未声明为真实正负例"
        print(f"{reason}，跳过混淆矩阵与准确率；仍导出分数和预测类别。")
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



