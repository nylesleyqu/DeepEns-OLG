from pathlib import Path
import hashlib
import json
import math
import random
import re
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ================= IDE 配置区 =================
MODE = "predict"
TRAIN_NPZ = Path(r"E:\olg\O-linkedGlycosylation\data\balanced_train_5\encoded_blosum62\train_balanced_05_blosum62.npz")
OUTPUT_ROOT = Path(r"E:\olg\O-linkedGlycosylation\data\blosum_local_adaptive_v2_pool_models")
RUN_NAME = "dataset_05_blosum_v2"
LOSS_MODE = "bce"
CONFIRMED_TRAIN_NEGATIVES = True  # 必填：真实负例=True；P/U=False且LOSS_MODE='nnpu'
PI_U = None
CLASSIFICATION_THRESHOLD = 0.5
EXPECTED_LENGTH = 21
CENTER_POSITION = 11
CHANNELS = 48
NUM_BLOCKS = 4
HEADS = 8
POSITION_DIM = 16
USE_LOCAL = True
USE_ATTENTION = True
READOUT = "pool"  # ordered:保序展开；pool:中心+均值+最大池化，用于消融
DROPOUT = 0.2
USE_SUBSTITUTION = True    # False:仅关闭注意力的BLOSUM弱偏置，保留节点编码和内容注意力
SELECTION_METRIC = "loss"  # loss或auc；nnpu只能loss。训练前确定，不根据测试集更换
SEED = 42
SPLIT_SEED = 42
INTERNAL_VAL_FRACTION = 0.2
EPOCHS = 120
PATIENCE = 15
MIN_DELTA = 1e-5
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0       # 先关闭以便解释损失；可独立消融
DEVICE = "auto"

CHECKPOINT = Path(r"E:\olg\O-linkedGlycosylation\data\blosum_local_adaptive_v2_pool_models\dataset_01_blosum_v2\best_model.pt")
PREDICT_NPZ = Path(r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_blosum62.npz")
PREDICT_OUTPUT = Path(r"E:\olg\O-linkedGlycosylation\data\blosum_local_adaptive_v2_pool_val_test\val_test_01.npz")
PREDICT_LABELS_ARE_TRUE_PN = True  # 有y时必填：真实P/N=True，P/U=False；无y可不填
# =============================================

ARCHITECTURE = "blosum_local_adaptive_ordered_v2"




class LocalAdaptiveBlock(nn.Module):
    """局部顺序建模 + 内容自适应注意力；BLOSUM仅作为弱偏置，不是接触图。"""
    def __init__(self, channels, length, heads=2, dropout=0.2,
                 use_local=True, use_attention=True, use_substitution=True):
        super().__init__()
        if channels % heads:
            raise ValueError("channels必须能被heads整除")
        self.heads, self.head_dim = heads, channels // heads
        self.use_local, self.use_attention = use_local, use_attention
        self.use_substitution = use_substitution and use_attention
        self.dropout = nn.Dropout(dropout)
        gate_init = math.log(0.1 / 0.9)
        if use_local:
            self.local_norm = nn.LayerNorm(channels)
            self.local = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                nn.GELU(), nn.Conv1d(channels, channels, 1))
            self.local_gate = nn.Parameter(torch.full((channels,), gate_init))
        if use_attention:
            self.attention_norm = nn.LayerNorm(channels)
            self.qkv = nn.Linear(channels, channels * 3, bias=False)
            self.out = nn.Linear(channels, channels, bias=False)
            positions = torch.arange(length)
            self.register_buffer("relative_index", positions[None, :] - positions[:, None] + length - 1)
            initial_bias = -torch.arange(-(length - 1), length).abs().float() / 8
            self.relative_bias = nn.Parameter(initial_bias.repeat(heads, 1))
            if self.use_substitution:
                # beta=0.5*tanh(raw)，初始化0.1，可减弱到0或改变符号。
                self.raw_prior_weight = nn.Parameter(torch.full((heads,), math.atanh(0.2)))
            self.attention_gate = nn.Parameter(torch.full((channels,), gate_init))
            self.ffn_norm = nn.LayerNorm(channels)
            self.ffn = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(2 * channels, channels))
            self.ffn_gate = nn.Parameter(torch.full((channels,), gate_init))

    def forward(self, h, pair_prior):
        if self.use_local:
            local = self.local(self.local_norm(h).transpose(1, 2)).transpose(1, 2)
            h = h + self.local_gate.sigmoid() * self.dropout(local)
        weights = None
        if self.use_attention:
            batch, length, channels = h.shape
            qkv = self.qkv(self.attention_norm(h)).reshape(
                batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            logits = logits + self.relative_bias[:, self.relative_index].unsqueeze(0)
            if self.use_substitution:
                beta = 0.5 * self.raw_prior_weight.tanh()
                logits = logits + beta.view(1, -1, 1, 1) * pair_prior.unsqueeze(1)
            weights = logits.softmax(-1)
            context = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, length, channels)
            h = h + self.attention_gate.sigmoid() * self.dropout(self.out(context))
            h = h + self.ffn_gate.sigmoid() * self.dropout(self.ffn(self.ffn_norm(h)))
        return h, weights


class LocalAdaptiveBLOSUM(nn.Module):
    """原始BLOSUM62 [B,L,20] + tokens [B,L] -> logits [B]。

    tokens 0..19对应encoding_matrix的行顺序；20表示未知残基。
    X是未知，不是padding：它参与卷积和内容注意力，但不提供BLOSUM边偏置。
    固定长度片段专用；若数据带真实padding，需另加独立padding mask。
    """
    def __init__(self, encoding_matrix, input_dim=20, length=21, center=10,
                 channels=24, num_blocks=1, heads=2, position_dim=8, dropout=0.2,
                 use_local=True, use_attention=True, use_substitution=True,
                 readout="ordered"):
        super().__init__()
        if input_dim != 20 or not 0 < center < length - 1:
            raise ValueError("需要20列BLOSUM编码，中心必须有左右上下文")
        if channels < 1 or num_blocks < 1 or heads < 1 or channels % heads:
            raise ValueError("channels/num_blocks/heads无效")
        if position_dim < 1 or not 0 <= dropout < 1 or readout not in {"ordered", "pool"}:
            raise ValueError("position_dim/dropout/readout无效")
        matrix = torch.as_tensor(encoding_matrix, dtype=torch.float32)
        if matrix.shape != (20, 20) or not torch.isfinite(matrix).all():
            raise ValueError("encoding_matrix必须为有限的20×20矩阵")
        self.length, self.center, self.input_dim, self.readout = length, center, input_dim, readout
        # 未知残基对应全零行列；有未知的残基对不提供替换先验。
        self.register_buffer("pair_prior_table", F.pad(matrix / matrix.abs().max().clamp_min(1), (0, 1, 0, 1)))
        self.register_buffer("profile_mean", matrix.mean(0))
        self.register_buffer("profile_std", matrix.std(0, unbiased=False).clamp_min(1e-6))
        self.projection = nn.Sequential(nn.Linear(input_dim * 2, channels), nn.GELU())
        self.unknown = nn.Parameter(torch.zeros(channels))
        self.position_embedding = nn.Parameter(torch.zeros(1, length, channels))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            LocalAdaptiveBlock(channels, length, heads, dropout,
                               use_local, use_attention, use_substitution)
            for _ in range(num_blocks)])
        self.output_norm = nn.LayerNorm(channels)
        if readout == "ordered":
            self.position_compress = nn.Sequential(nn.Linear(channels, position_dim), nn.GELU())
            head_dim = length * position_dim + channels
        else:
            head_dim = channels * 3
        self.head = nn.Sequential(nn.Linear(head_dim, 32), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x, tokens, return_weights=False):
        if (x.ndim != 3 or x.shape[1:] != (self.length, self.input_dim)
                or tokens.shape != x.shape[:2]):
            raise ValueError("输入必须为[B,L,20]，tokens必须为[B,L]")
        if tokens.dtype != torch.long or torch.any((tokens < 0) | (tokens > 20)):
            raise ValueError("tokens须为long且在0..20之间")
        x = x.to(self.profile_mean.dtype)
        known = tokens < 20
        z = ((x - self.profile_mean) / self.profile_std) * known.unsqueeze(-1)
        delta_mask = known & known[:, self.center:self.center+1]
        delta = (z - z[:, self.center:self.center+1]) * delta_mask.unsqueeze(-1)
        h = self.projection(torch.cat((z, delta), dim=-1))
        h = torch.where(known.unsqueeze(-1), h, self.unknown.view(1, 1, -1))
        h = self.input_dropout(h + self.position_embedding)
        pair_prior = self.pair_prior_table[tokens.unsqueeze(2), tokens.unsqueeze(1)]
        weights = None
        for block in self.blocks:
            h, weights = block(h, pair_prior)
        h = self.output_norm(h)
        if self.readout == "ordered":
            # 按原位置顺序展开，不做全局池化；再次单独保留中心特征。
            features = torch.cat((self.position_compress(h).flatten(1), h[:, self.center]), dim=-1)
        else:
            features = torch.cat((h[:, self.center], h.mean(1), h.max(1).values), dim=-1)
        logits = self.head(features).squeeze(-1)
        return (logits, weights) if return_weights else logits

    @torch.no_grad()
    def diagnostics(self):
        result = []
        for block in self.blocks:
            item = {}
            if block.use_local:
                item["local_gate_mean"] = block.local_gate.sigmoid().mean().item()
            if block.use_attention:
                item["attention_gate_mean"] = block.attention_gate.sigmoid().mean().item()
                item["ffn_gate_mean"] = block.ffn_gate.sigmoid().mean().item()
            if block.use_substitution:
                item["prior_weights"] = (0.5 * block.raw_prior_weight.tanh()).cpu().tolist()
            result.append(item)
        return result

def load_data(path, require_labels=True):
    with np.load(path, allow_pickle=False) as f:
        required = {"X", "ids", "sequences", "known_residue_mask", "feature_columns", "encoding_matrix"}
        missing = required - set(f.files)
        if missing:
            raise ValueError(f"{path} 缺少 {sorted(missing)}；需要原始BLOSUM62编码NPZ，不是预测结果")
        data = {k: f[k].copy() for k in f.files}
    x = np.asarray(data["X"], dtype=np.float32)
    ids = data["ids"].astype(str)
    columns = data["feature_columns"].astype(str)
    matrix = np.asarray(data["encoding_matrix"], dtype=np.float32)
    sequences = data["sequences"].astype(str)
    if x.ndim != 3 or min(x.shape) < 1 or x.shape[2] != 20 or not np.isfinite(x).all():
        raise ValueError("X必须为非空有限[N,L,20]数组")
    if columns.shape != (20,) or set(columns.tolist()) != set("ARNDCQEGHILKMFPSTWYV"):
        raise ValueError("feature_columns须是20种标准氨基酸，不重复")
    if matrix.shape != (20, 20) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError("encoding_matrix必须为有限且对称的20×20替换得分矩阵")
    if ids.shape != (len(x),) or len(set(ids.tolist())) != len(ids) or any(not s for s in ids):
        raise ValueError("ID为空、重复或数量错误")
    if sequences.shape != ids.shape or any(len(s) != x.shape[1] for s in sequences):
        raise ValueError("sequences数量或片段长度错误")
    lookup = {a: i for i, a in enumerate(columns)}
    unknowns = set("XUBZOJ-")
    if any(a not in lookup and a not in unknowns for s in sequences for a in s):
        raise ValueError("sequences存在不支持的字符，请使用配套编码程序")
    tokens = np.asarray([[lookup.get(a, 20) for a in s] for s in sequences], dtype=np.int64)
    mask = data["known_residue_mask"]
    if mask.shape != x.shape[:2] or not np.isin(mask, [0, 1]).all() or not np.array_equal(mask.astype(bool), tokens < 20):
        raise ValueError("known_residue_mask与sequences不一致")
    table = np.concatenate((matrix, np.zeros((1, 20), dtype=np.float32)), axis=0)
    if not np.allclose(x, table[tokens], atol=1e-6, rtol=0):
        raise ValueError("X与sequences/encoding_matrix不一致；需要原始得分，不接受预标准化X")
    proteins = data.get("protein_ids")
    if proteins is None:
        if any(not re.fullmatch(r".+_\d+", identifier) for identifier in ids):
            raise ValueError("需要protein_ids，或ID采用 蛋白质名_数字位点 格式")
        proteins = np.asarray([identifier.rsplit("_", 1)[0] for identifier in ids])
    proteins = proteins.astype(str)
    if proteins.shape != ids.shape or any(not p for p in proteins):
        raise ValueError("protein_ids无效")
    center_1 = int(data.get("center_position_1_based", np.asarray(CENTER_POSITION)).item())
    if x.shape[1] != EXPECTED_LENGTH or center_1 != CENTER_POSITION or not 1 < center_1 < x.shape[1]:
        raise ValueError("片段长度/中心位置与配置不匹配")
    y = data.get("y")
    if y is None and require_labels:
        raise ValueError("训练文件缺少y；编码时必须提供标签表")
    if y is not None:
        if y.shape != ids.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("y须为逐样本0/1标签")
        y = y.astype(np.int64)
    schema = {"encoding": "blosum62_raw", "feature_columns": columns.tolist(),
              "matrix_sha256": hashlib.sha256(matrix.astype("<f4").tobytes()).hexdigest(),
              "length": x.shape[1], "input_dim": 20, "center": center_1 - 1}
    semantics = str(data.get("label_semantics", np.asarray("unspecified")).item())
    return {"X": x, "tokens": tokens, "encoding_matrix": matrix.tolist(),
            "ids": ids, "proteins": proteins, "y": y, "schema": schema,
            "source_label_semantics": semantics}

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
        tokens = torch.from_numpy(data["tokens"][ix]).to(device)
        results.append(model(x, tokens).cpu().numpy())
    result = np.concatenate(results)
    if not np.isfinite(result).all():
        raise RuntimeError("预测出现 NaN/Inf")
    return result



def average_precision(logits, labels):
    """按分数并列组计算阶梯PR曲线的AP，非梯形面积；无正例时返回None。"""
    scores = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    scores, y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(y) - 1]
    tp = np.cumsum(y)[ends].astype(np.float64)
    precision = tp / (ends + 1)
    increments = np.diff(np.r_[0., tp]) / positives
    return float(np.sum(increments * precision))

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
            "f1": f1, "balanced_accuracy": balanced, "average_precision": average_precision(logits, y),
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
        payload["average_precision"] = np.asarray(np.nan if metrics["average_precision"] is None else metrics["average_precision"])
        payload["label_order"] = np.asarray([0, 1])
        payload["matrix_axes"] = np.asarray(metrics["matrix_axes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)


def train():
    if CONFIRMED_TRAIN_NEGATIVES is not True and CONFIRMED_TRAIN_NEGATIVES is not False:
        raise ValueError("请先明确标签：真实负例设CONFIRMED_TRAIN_NEGATIVES=True、LOSS_MODE='bce'；"
                         "未注释候选位点设False、LOSS_MODE='nnpu'并估计PI_U。")
    if LOSS_MODE == "bce" and not CONFIRMED_TRAIN_NEGATIVES:
        raise ValueError("BCE要求真实负例；未注释候选位点请使用nnpu并估计PI_U")
    if LOSS_MODE == "nnpu" and CONFIRMED_TRAIN_NEGATIVES:
        raise ValueError("真实负例请使用bce；nnpu要求0为未注释候选位点")
    if SELECTION_METRIC not in {"loss", "auc"} or (LOSS_MODE == "nnpu" and SELECTION_METRIC != "loss"):
        raise ValueError("SELECTION_METRIC只能为loss/auc；nnpu不能用P/U分类AUC选模型")
    if MIN_DELTA < 0 or not math.isfinite(MIN_DELTA):
        raise ValueError("MIN_DELTA应是非负有限数")
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
    data = load_data(TRAIN_NPZ)
    print(f"输入文件标签语义：{data['source_label_semantics']}")
    if LOSS_MODE == "bce":
        print("注意：本次按用户配置将0视作真实负例；不会自动核实或改变原始注释。")
    tr, va = split_groups(data["proteins"], data["y"], INTERNAL_VAL_FRACTION, SPLIT_SEED)
    device = get_device()
    schema = data["schema"]
    kwargs = {k: schema[k] for k in ("input_dim", "length", "center")}
    kwargs.update(encoding_matrix=data["encoding_matrix"], channels=CHANNELS,
                  num_blocks=NUM_BLOCKS, heads=HEADS, position_dim=POSITION_DIM,
                  dropout=DROPOUT, use_local=USE_LOCAL, use_attention=USE_ATTENTION,
                  use_substitution=USE_SUBSTITUTION, readout=READOUT)
    model = LocalAdaptiveBLOSUM(**kwargs).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, threshold=MIN_DELTA,
        threshold_mode="abs", min_lr=LEARNING_RATE * 0.02)

    output = OUTPUT_ROOT / f"{RUN_NAME}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best_model.pt"
    config = {"train_npz": str(TRAIN_NPZ.resolve()), "loss_mode": LOSS_MODE,
              "pi_u": PI_U if LOSS_MODE == "nnpu" else None,
              "classification_threshold": CLASSIFICATION_THRESHOLD,
              "selection_metric": SELECTION_METRIC if LOSS_MODE == "bce" else "nnpu_risk",
              "label_smoothing": LABEL_SMOOTHING,
              "scheduler": "ReduceLROnPlateau(factor=0.5, patience=4)",
              "seed": SEED, "split_seed": SPLIT_SEED, "internal_val_fraction": INTERNAL_VAL_FRACTION,
              "epochs": EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
              "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
              "architecture": ARCHITECTURE, "parameters": parameter_count,
              "device": str(device), "torch_version": str(torch.__version__),
              "label_semantics": "1=P,0=U" if LOSS_MODE == "nnpu" else "1=P,0=verified N",
              "schema": schema, "model_kwargs": kwargs,
              "source_label_semantics": data["source_label_semantics"],
              "confirmed_train_negatives": CONFIRMED_TRAIN_NEGATIVES,
              "sampling": "shuffle once per epoch without replacement" if LOSS_MODE == "bce" else "balanced PU with replacement",
              "note": "internal validation predictions are NOT independent stacking training features"}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output / "split_ids.npz", train_ids=data["ids"][tr], val_ids=data["ids"][va],
                        train_proteins=data["proteins"][tr], val_proteins=data["proteins"][va])
    print(f"设备={device}；参数={parameter_count:,}；内部训练={len(tr)}；内部验证={len(va)}")
    if LOSS_MODE == "bce":
        print(f"BCE 二分类：1=正例，0=真实负例；准确率阈值={CLASSIFICATION_THRESHOLD}；按预先设置的{SELECTION_METRIC}选择模型。")
    else:
        print(f"nnPU：1=已标注正例，0=未标注候选位点；PI_U={PI_U}；不把U当N报告分类准确率。")
    for name, rows in (("内部训练", tr), ("内部验证", va)):
        count = int(data["y"][rows].sum())
        print(f"{name}：标签1={count}，标签0={len(rows)-count}，多数类基线={max(count,len(rows)-count)/len(rows):.2%}")
    print(f"输出目录：{output}")
    pools = [tr[data["y"][tr] == label] for label in (1, 0)]
    steps = math.ceil(len(tr) / BATCH_SIZE)
    best, stale, history = float("inf"), 0, []
    best_loss, best_auc = float("inf"), -float("inf")
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
            tokens = torch.from_numpy(data["tokens"][rows]).to(device)
            objective = training_loss(model(x, tokens), y, LOSS_MODE, PI_U,
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
                        "lr": current_lr, "model_diagnostics": model.diagnostics()})
        selection_value = risk if SELECTION_METRIC == "loss" else -auc
        improved = selection_value < best - MIN_DELTA
        loss_improved = risk < best_loss
        auc_improved = auc is not None and auc > best_auc
        if improved or loss_improved or auc_improved:
            state = {"state_dict": model.state_dict(), "model_kwargs": kwargs,
                     "schema": schema, "config": config, "epoch": epoch, "val_risk": risk,
                     "val_accuracy": accuracy, "val_auc": auc, "val_mcc": mcc,
                     "seen_proteins": sorted(set(data["proteins"].tolist())),
                     "seen_ids": data["ids"].tolist()}
            if loss_improved:
                best_loss = risk
                torch.save(dict(state, checkpoint_kind="best_loss"), output / "best_loss.pt")
            if auc_improved:
                best_auc = auc
                torch.save(dict(state, checkpoint_kind="best_auc"), output / "best_auc.pt")
            if improved:
                torch.save(dict(state, checkpoint_kind="selected_" + SELECTION_METRIC), checkpoint)
        if improved:
            best, stale = selection_value, 0
        else:
            stale += 1
        scheduler.step(selection_value)

        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        progress = (f"训练损失={update_mean:.6f} | 训练评估损失={train_risk:.6f} | 内部验证损失={risk:.6f}"
                    f" | 训练准确率={train_accuracy:.2%} | 验证准确率={accuracy:.2%} | AUC={auc:.4f} | MCC={mcc:.4f}"
                    f" | AP={metrics['average_precision']:.4f} | 平衡准确率={metrics['balanced_accuracy']:.2%} | 召回率={metrics['recall']:.2%} | lr={current_lr:.2e}"
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
    if PREDICT_OUTPUT.resolve() in {PREDICT_NPZ.resolve(), CHECKPOINT.resolve()}:
        raise ValueError("输出不能覆盖输入")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if saved.get("config", {}).get("architecture") != ARCHITECTURE:
        raise ValueError(f"请加载本脚本({ARCHITECTURE})生成的权重")
    data = load_data(PREDICT_NPZ, require_labels=False)
    threshold = float(saved.get("config", {}).get("classification_threshold", CLASSIFICATION_THRESHOLD))
    if not 0 < threshold < 1:
        raise ValueError("模型保存的分类阈值必须在 (0,1)")
    if data["y"] is not None and PREDICT_LABELS_ARE_TRUE_PN is not True and PREDICT_LABELS_ARE_TRUE_PN is not False:
        raise ValueError("预测文件有y，请明确PREDICT_LABELS_ARE_TRUE_PN=True(真实P/N)或False(P/U)。")
    evaluate = data["y"] is not None and PREDICT_LABELS_ARE_TRUE_PN
    metrics_path = PREDICT_OUTPUT.with_name(PREDICT_OUTPUT.stem + "_metrics.json")
    if evaluate:
        if metrics_path.exists():
            raise FileExistsError(f"拒绝覆盖已有评估结果：{metrics_path}")
        if metrics_path.resolve() in {PREDICT_NPZ.resolve(), CHECKPOINT.resolve()}:
            raise ValueError("评估结果路径不能覆盖输入")
    if data["schema"] != saved["schema"]:
        raise ValueError("编码不匹配：请核对BLOSUM矩阵、通道顺序、长度及中心位置")
    overlap = set(data["proteins"].tolist()) & set(saved["seen_proteins"])
    if overlap:
        raise ValueError(f"预测集与基层训练/早停数据重叠 {len(overlap)} 个蛋白质，拒绝生成独立融合特征")
    device = get_device()
    model = LocalAdaptiveBLOSUM(**saved["model_kwargs"]).to(device)
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
        print(f"预测分类阈值={threshold:g}")
        print("混淆矩阵：行=真实类别，列=预测类别，0=负例，1=正例")
        print("                  预测0       预测1")
        print(f"真实0（负例） {metrics['tn']:10d}  {metrics['fp']:10d}")
        print(f"真实1（正例） {metrics['fn']:10d}  {metrics['tp']:10d}")
        print(f"准确率={metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['sample_count']})"
              f" | AUC={metrics['auc']} | AP={metrics['average_precision']} | MCC={metrics['mcc']:.4f}")
        print(f"评估结果：{metrics_path}")
    else:
        reason = "预测文件没有 y 标签" if data["y"] is None else "标签未声明为真实正负例"
        print(f"{reason}，跳过混淆矩阵与准确率；仍导出分数和预测类别。")
    print("与其他编码模型融合前必须按 ids 对齐；本程序未训练外层 MLP。")
    print("所有子集/编码模型应采用事先确定的检查点选择规则；不可按测试表现选择best_loss/best_auc。")


def main():
    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise ValueError("MODE 只能为 train 或 predict")


if __name__ == "__main__":
    main()



