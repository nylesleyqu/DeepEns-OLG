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
TRAIN_NPZ = Path(r"E:\olg\O-linkedGlycosylation\data\balanced_train_5\encoded_onehot\train_balanced_01_onehot.npz")
OUTPUT_ROOT = Path(r"E:\olg\O-linkedGlycosylation\data\onehot_bilstm_internal_residual_models")
RUN_NAME = "dataset_01_internal_residual"
LOSS_MODE = "bce"
PI_U = None
CLASSIFICATION_THRESHOLD = 0.5
EXPECTED_LENGTH = 21
CENTER_POSITION = 11
EMBEDDING_DIM = 64          # 16 -> 32
HIDDEN_SIZE = 64            # 32 -> 64
DROPOUT = 0.2              # 稍大以抵消容量提升
RESIDUAL_SCALE = 1.0
SEED = 42
SPLIT_SEED = 42
INTERNAL_VAL_FRACTION = 0.2
EPOCHS = 120
PATIENCE = 20
MIN_DELTA = 1e-5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.02      # 轻微平滑；0 关闭
DEVICE = "auto"

CHECKPOINT = Path(r"E:\olg\O-linkedGlycosylation\data\onehot_bilstm_internal_residual_models\dataset_05_internal_residual\best_model.pt")
PREDICT_NPZ = Path(r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_onehot.npz")
PREDICT_OUTPUT = Path(r"E:\olg\O-linkedGlycosylation\data\onehot_bilstm_internal_residual_val_test\val_test_05.npz")
PREDICT_LABELS_ARE_TRUE_PN = True
# =============================================

ARCHITECTURE = "onehot_bilstm_internal_residual_v2"


class InternalResidualCell(nn.Module):
    """输入投影残差进入 h_t，且修改后的 h_t 参与下一步门控。c_t 仍标准 LSTM 更新。"""
    def __init__(self, input_size, hidden_size, residual_scale=1.0):
        super().__init__()
        self.base = nn.LSTMCell(input_size, hidden_size)
        self.skip = nn.Linear(input_size, hidden_size, bias=False)
        self.residual_gate = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, math.log(0.1 / 0.9))
        self.residual_scale = residual_scale

    def forward(self, x, state):
        previous_h, previous_c = state
        base_h, c = self.base(x, (previous_h, previous_c))
        gate = torch.sigmoid(self.residual_gate(torch.cat((x, previous_h), dim=-1)))
        residual = torch.tanh(self.skip(x))
        h = base_h + self.residual_scale * gate * residual
        return h, c


class InternalResidualBiLSTM(nn.Module):
    """显式展开两个方向；反向结果恢复原位置顺序后再拼接。"""
    def __init__(self, input_size, hidden_size, residual_scale=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.forward_cell = InternalResidualCell(input_size, hidden_size, residual_scale)
        self.backward_cell = InternalResidualCell(input_size, hidden_size, residual_scale)

    def forward(self, x):
        batch, length, _ = x.shape
        h_f = x.new_zeros(batch, self.hidden_size)
        c_f = x.new_zeros(batch, self.hidden_size)
        h_b = x.new_zeros(batch, self.hidden_size)
        c_b = x.new_zeros(batch, self.hidden_size)
        forward_outputs, backward_outputs = [], []
        for position in range(length):
            h_f, c_f = self.forward_cell(x[:, position], (h_f, c_f))
            forward_outputs.append(h_f)
        for position in range(length - 1, -1, -1):
            h_b, c_b = self.backward_cell(x[:, position], (h_b, c_b))
            backward_outputs.append(h_b)
        output = torch.cat((torch.stack(forward_outputs, dim=1),
                            torch.stack(backward_outputs[::-1], dim=1)), dim=-1)
        return output, (torch.stack((h_f, h_b)), torch.stack((c_f, c_b)))


class InternalResidualOneHotBiLSTM(nn.Module):
    """单层双向 LSTM + 单元内部受门控残差 + 多路池化融合。

    池化融合：左均值 / 中心 / 右均值 / 注意力池化 / 全局最大池化。
    """
    def __init__(
        self,
        input_dim=21,
        length=21,
        center=10,
        embedding_dim=64,
        hidden_size=64,
        dropout=0.1,
        residual_scale=1.0,
    ):
        super().__init__()
        if not 0 < center < length - 1:
            raise ValueError("中心位置必须同时具有左右上下文")
        self.input_dim = input_dim
        self.length = length
        self.center = center

        # 独热 -> 可学习残基嵌入
        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        # 位置嵌入，形状 [1, L, E]，广播到 batch
        self.position_embedding = nn.Parameter(torch.zeros(1, length, embedding_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.input_dropout = nn.Dropout(dropout)

        if not 0 <= residual_scale <= 1:
            raise ValueError("residual_scale应在[0,1]，0用于关闭内部残差的消融实验")
        self.lstm = InternalResidualBiLSTM(embedding_dim, hidden_size, residual_scale)
        context_dim = hidden_size * 2

        # 注意力池化：位置打分
        self.attn_pool = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.Tanh(),
            nn.Linear(context_dim, 1),
        )
        pooled_dim = context_dim * 5  # 左 / 中心 / 右 / 注意力 / 最大

        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("输入必须为 [B, L, A]")
        if x.shape[1] != self.length:
            raise ValueError(f"片段长度应为 {self.length}，实际为 {x.shape[1]}")
        if x.shape[2] != self.input_dim:
            raise ValueError(f"编码通道数应为 {self.input_dim}，实际为 {x.shape[2]}")

        x = x.to(dtype=self.embedding.weight.dtype)
        embedded = self.embedding(x) + self.position_embedding  # [B,L,E]
        embedded = self.input_dropout(embedded)
        output, _ = self.lstm(embedded)  # [B,L,2H]

        c = self.center
        left_context = output[:, :c, :].mean(dim=1)
        center_context = output[:, c, :]
        right_context = output[:, c + 1:, :].mean(dim=1)

        attn_scores = self.attn_pool(output)                  # [B,L,1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_context = (output * attn_weights).sum(dim=1)     # [B,2H]

        max_context = output.max(dim=1).values                # [B,2H]

        features = torch.cat(
            [left_context, center_context, right_context, attn_context, max_context],
            dim=-1,
        )
        features = self.output_dropout(features)
        return self.head(features).squeeze(-1)


def load_data(path, require_labels=True):
    with np.load(path, allow_pickle=False) as f:
        missing = {"X", "ids", "alphabet"} - set(f.files)
        if missing:
            raise ValueError(f"{path} 缺少 {sorted(missing)}，请提供独热编码NPZ，不是预测结果")
        data = {k: f[k].copy() for k in f.files}
    x = data["X"]
    ids = data["ids"].astype(str)
    alphabet = data["alphabet"].astype(str)
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError("X 必须为非空 [N,L,A]")
    if not np.isin(x, [0, 1]).all() or not np.all(x.sum(axis=-1) == 1):
        raise ValueError("每个位置必须恰有一个1，其余为0；不支持全零padding或软编码")
    if alphabet.shape != (x.shape[2],) or set(alphabet.tolist()) != set("ACDEFGHIKLMNPQRSTVWYX"):
        raise ValueError("alphabet必须包含20种标准氨基酸和X，且与通道数一致")
    if ids.shape != (len(x),) or len(set(ids.tolist())) != len(ids) or any(not s for s in ids):
        raise ValueError("样本ID为空、重复或数量错误")
    proteins = data.get("protein_ids")
    if proteins is None:
        if any(not re.fullmatch(r".+_\d+", identifier) for identifier in ids):
            raise ValueError("需要protein_ids，或ID采用 蛋白质名_数字位点 格式")
        proteins = np.asarray([identifier.rsplit("_", 1)[0] for identifier in ids])
    proteins = proteins.astype(str)
    if proteins.shape != ids.shape or any(not p for p in proteins):
        raise ValueError("protein_ids无效")
    center_1 = int(data.get("center_position_1_based", np.asarray(CENTER_POSITION)).item())
    if x.shape[1] != EXPECTED_LENGTH or center_1 != CENTER_POSITION:
        raise ValueError("编码长度或中心位置与配置不符，请检查EXPECTED_LENGTH和CENTER_POSITION")
    if not 1 < center_1 < x.shape[1]:
        raise ValueError("中心必须同时具有左右上下文")
    if "sequences" in data:
        sequences = data["sequences"].astype(str)
        if sequences.shape != ids.shape:
            raise ValueError("sequences数量不符")
        decoded = ["".join(alphabet[row]) for row in x.argmax(-1)]
        if any(a != b for a, b in zip(sequences, decoded)):
            raise ValueError("X与sequences/alphabet不一致")
    y = data.get("y")
    if y is None and require_labels:
        raise ValueError("训练文件缺少y，编码时需要提供标签表")
    if y is not None:
        if y.shape != ids.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("y必须为逐样本0/1标签")
        y = y.astype(np.int64)
    schema = {"encoding": "one_hot", "alphabet": alphabet.tolist(),
              "length": x.shape[1], "input_dim": x.shape[2], "center": center_1 - 1}
    return {"X": x.astype(np.uint8), "ids": ids, "proteins": proteins,
            "y": y, "schema": schema}


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
        return float("nan")
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
    return {"accuracy": accuracy, "correct_count": int(np.trace(matrix)),
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
        payload["auc"] = np.asarray(metrics["auc"])
        payload["mcc"] = np.asarray(metrics["mcc"])
        payload["label_order"] = np.asarray([0, 1])
        payload["matrix_axes"] = np.asarray(metrics["matrix_axes"])
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
    kwargs.update(embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
                  dropout=DROPOUT, residual_scale=RESIDUAL_SCALE)
    model = InternalResidualOneHotBiLSTM(**kwargs).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 余弦退火：单调递减到 lr*0.02
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.02)

    output = OUTPUT_ROOT / f"{RUN_NAME}"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best_model.pt"
    config = {"train_npz": str(TRAIN_NPZ.resolve()), "loss_mode": LOSS_MODE,
              "pi_u": PI_U if LOSS_MODE == "nnpu" else None,
              "classification_threshold": CLASSIFICATION_THRESHOLD,
              "selection_metric": "val_loss" if LOSS_MODE == "bce" else "val_risk",
              "label_smoothing": LABEL_SMOOTHING,
              "scheduler": f"CosineAnnealingLR(T_max={EPOCHS}, eta_min={LEARNING_RATE * 0.02:g})",
              "seed": SEED, "split_seed": SPLIT_SEED, "internal_val_fraction": INTERNAL_VAL_FRACTION,
              "epochs": EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
              "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
              "architecture": ARCHITECTURE, "parameters": parameter_count,
              "device": str(device), "torch_version": str(torch.__version__),
              "label_semantics": "1=P,0=U" if LOSS_MODE == "nnpu" else "1=P,0=verified N",
              "schema": schema,
              "sampling": "equal label-1/label-0 batches; uniform replacement within each pool",
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

        logits = predict_logits(model, data, va, device)
        risk = validation_risk(logits, data["y"][va], LOSS_MODE, PI_U)
        y_va = data["y"][va]
        accuracy = validation_accuracy(logits, y_va, CLASSIFICATION_THRESHOLD)
        auc = compute_auc(logits, y_va)
        mcc = compute_mcc(logits, y_va, CLASSIFICATION_THRESHOLD)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_update_objective": float(np.mean(objectives)),
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
        scheduler.step()

        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        progress = (f"训练损失={np.mean(objectives):.6f} | 内部验证损失={risk:.6f}"
                    f" | 准确率={accuracy:.2%} | AUC={auc:.4f} | MCC={mcc:.4f} | lr={current_lr:.2e}"
                    if LOSS_MODE == "bce" else
                    f"更新目标={np.mean(objectives):.6f} | 内部验证风险={risk:.6f}"
                    f" | AUC={auc:.4f} | MCC={mcc:.4f} | lr={current_lr:.2e}")
        print(f"Epoch {epoch:03d} | {progress}" + (" | 保存最佳权重" if improved else ""))
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
    evaluate = data["y"] is not None and PREDICT_LABELS_ARE_TRUE_PN
    metrics_path = PREDICT_OUTPUT.with_name(PREDICT_OUTPUT.stem + "_metrics.json")
    if evaluate:
        if metrics_path.exists():
            raise FileExistsError(f"拒绝覆盖已有评估结果：{metrics_path}")
        if metrics_path.resolve() in {PREDICT_NPZ.resolve(), CHECKPOINT.resolve()}:
            raise ValueError("评估结果路径不能覆盖输入")
    if data["schema"] != saved["schema"]:
        raise ValueError("编码不匹配：请核对独热通道顺序、长度及中心位置")
    overlap = set(data["proteins"].tolist()) & set(saved["seen_proteins"])
    if overlap:
        raise ValueError(f"预测集与基层训练/早停数据重叠 {len(overlap)} 个蛋白质，拒绝生成独立融合特征")
    device = get_device()
    model = InternalResidualOneHotBiLSTM(**saved["model_kwargs"]).to(device)
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
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"预测分类阈值={threshold:g}")
        print("混淆矩阵：行=真实类别，列=预测类别，0=负例，1=正例")
        print("                  预测0       预测1")
        print(f"真实0（负例） {metrics['tn']:10d}  {metrics['fp']:10d}")
        print(f"真实1（正例） {metrics['fn']:10d}  {metrics['tp']:10d}")
        print(f"准确率={metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['sample_count']})"
              f" | AUC={metrics['auc']:.4f} | MCC={metrics['mcc']:.4f}")
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