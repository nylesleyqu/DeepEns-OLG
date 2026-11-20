#!/usr/bin/env python3
"""对单个FASTA文件进行BLOSUM62逐残基编码，可在IDE中直接运行。

依赖：numpy、biopython。
每次只处理INPUT_FASTA指定的一个文件，并保持FASTA样本顺序不变。
LABELS_TSV可选：提供时按sample_id对齐并保存y；为None时不猜测标签。
X 和 - 可作为未知/缺失占位符：保持 BLOSUM62 零向量编码，并额外保存不同掩码，
避免它们被错误解释为某种标准氨基酸。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    from Bio import SeqIO
    from Bio.Align import substitution_matrices
except ImportError as exc:
    raise SystemExit(
        "请在IDE使用的Python环境安装依赖：python -m pip install numpy biopython"
    ) from exc


# ========================== IDE运行配置区 ==========================
INPUT_FASTA = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test.fasta"
)

# 混合P/U文件可配置配套标签表，表中必须有sample_id和label两列。
# 若只需要编码、或输入FASTA没有可靠标签，请设置为None；程序不会根据文件名猜标签。
LABELS_TSV: Path | None = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_label.tsv"
)

OUTPUT_FILE = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_blosum62.npz"
)
METADATA_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_metadata.json")

FRAGMENT_LENGTH = 21
ALLOW_OVERWRITE = False

# X 与 - 是允许的占位符。二者在 BLOSUM62 主矩阵中均保留全零向量，区别写入
# placeholder_type / x_placeholder_mask / gap_placeholder_mask，不改变 X 的形状。
# B/Z/U/O/J 等无法唯一映射的字符会报错，避免静默猜测替换。
PLACEHOLDER_SYMBOLS = frozenset({"X", "-"})
PLACEHOLDER_POLICY = "zero_with_mask"  # zero_with_mask 或 error
# ==================================================================


# 固定列顺序；后续模型必须读取feature_columns确认，不能自行改变。
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
# X/- 是占位符，不是 BLOSUM62 的真实残基行。
UNKNOWN_RESIDUES = PLACEHOLDER_SYMBOLS
LABEL_SEMANTICS = "1=labeled_positive(P); 0=unlabeled(U), NOT verified_negative"

BLOSUM62 = substitution_matrices.load("BLOSUM62")
ENCODING_MATRIX = np.asarray(
    [[BLOSUM62[residue, column] for column in AMINO_ACIDS]
     for residue in AMINO_ACIDS],
    dtype=np.float32,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """读取一个FASTA文件，支持多行序列并检查ID和长度。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{path}")
    sample_ids, sequences = [], []
    with path.open("r", encoding="utf-8-sig") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sample_id = record.id
            sequence = str(record.seq).upper()
            if not sample_id:
                raise ValueError(f"{path}中存在空样本ID")
            if not sequence:
                raise ValueError(f"{path}中的{sample_id}为空序列")
            if len(sequence) != FRAGMENT_LENGTH:
                raise ValueError(
                    f"样本{sample_id}长度为{len(sequence)}，预期为{FRAGMENT_LENGTH}；"
                    "程序不会自动截断或补齐"
                )
            invalid = set(sequence) - set(AMINO_ACIDS) - set(PLACEHOLDER_SYMBOLS)
            placeholders = set(sequence) & set(PLACEHOLDER_SYMBOLS)
            if invalid:
                raise ValueError(
                    f"样本{sample_id}含不支持的字符：{sorted(invalid)}；仅允许标准氨基酸和"
                    f"占位符{sorted(PLACEHOLDER_SYMBOLS)}。B/Z/U/O/J不能自动猜测替换。"
                )
            if placeholders and PLACEHOLDER_POLICY == "error":
                raise ValueError(
                    f"样本{sample_id}含占位符{sorted(placeholders)}，但PLACEHOLDER_POLICY='error'"
                )
            sample_ids.append(sample_id)
            sequences.append(sequence)
    if not sample_ids:
        raise ValueError(f"{path}中没有FASTA记录")
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items()
                  if count > 1]
    if duplicates:
        raise ValueError(f"{path}中存在重复样本ID：{duplicates[:5]}")
    return sample_ids, sequences


def read_labels(path: Path, fasta_ids: list[str]) -> np.ndarray:
    """读取TSV并按sample_id对齐；标签表顺序可以与FASTA不同。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到标签文件：{path}")
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"sample_id", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}必须包含sample_id和label两列")
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not sample_id:
                raise ValueError(f"{path}第{line_number}行sample_id为空")
            if sample_id in mapping:
                raise ValueError(f"{path}存在重复sample_id：{sample_id}")
            if label_text not in {"0", "1"}:
                raise ValueError(f"{path}第{line_number}行label必须是0或1")
            mapping[sample_id] = int(label_text)
    missing = [sample_id for sample_id in fasta_ids if sample_id not in mapping]
    extra = sorted(set(mapping) - set(fasta_ids))
    if missing or extra:
        raise ValueError(
            f"FASTA与标签表ID不完全一致：缺少标签{len(missing)}个，"
            f"标签表多出{len(extra)}个；示例missing={missing[:3]}, extra={extra[:3]}"
        )
    return np.asarray([mapping[sample_id] for sample_id in fasta_ids], dtype=np.uint8)


def encode_blosum62(sequences: list[str]):
    """返回 BLOSUM62 特征及 X/- 的显式掩码。

    placeholder_type：0=标准氨基酸，1=X，2=-。X 和 - 都保留全零 BLOSUM62 向量，
    但其来源通过 placeholder_type 区分；既有模型仍可仅使用 known_residue_mask。
    """
    if PLACEHOLDER_POLICY not in {"zero_with_mask", "error"}:
        raise ValueError("PLACEHOLDER_POLICY只能是 'zero_with_mask' 或 'error'")
    sample_count = len(sequences)
    features = np.zeros(
        (sample_count, FRAGMENT_LENGTH, len(AMINO_ACIDS)), dtype=np.float32
    )
    known_mask = np.zeros((sample_count, FRAGMENT_LENGTH), dtype=np.uint8)
    placeholder_type = np.zeros((sample_count, FRAGMENT_LENGTH), dtype=np.uint8)
    normalized_sequences, placeholder_counts = [], Counter()
    placeholder_codes = {"X": 1, "-": 2}
    for sample_index, sequence in enumerate(sequences):
        normalized = []
        for position, residue in enumerate(sequence):
            if residue in AA_TO_INDEX:
                features[sample_index, position] = ENCODING_MATRIX[AA_TO_INDEX[residue]]
                known_mask[sample_index, position] = 1
                normalized.append(residue)
            elif residue in placeholder_codes:
                if PLACEHOLDER_POLICY == "error":
                    raise ValueError(f"发现占位符{residue!r}，但PLACEHOLDER_POLICY='error'")
                # 保持零向量；不使用 BLOSUM62 中字符 X 的替代评分行。
                placeholder_type[sample_index, position] = placeholder_codes[residue]
                normalized.append("X")  # 保持旧 sequences 字段的兼容性
                placeholder_counts[residue] += 1
            else:
                raise ValueError(f"不支持的残基{residue!r}；请先通过read_fasta验证")
        normalized_sequences.append("".join(normalized))
    return features, known_mask, normalized_sequences, placeholder_type, placeholder_counts


def atomic_save_npz(path: Path, **arrays) -> None:
    """先写临时文件再替换目标，避免异常时留下半个NPZ。"""
    if path.exists() and not ALLOW_OVERWRITE:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    if FRAGMENT_LENGTH < 1:
        raise ValueError("FRAGMENT_LENGTH必须为正整数")
    existing = [str(path) for path in (OUTPUT_FILE, METADATA_FILE) if path.exists()]
    if existing and not ALLOW_OVERWRITE:
        raise FileExistsError(
            "已有输出；请修改OUTPUT_FILE，或确认后设置ALLOW_OVERWRITE=True：\n"
            + "\n".join(existing)
        )
    input_paths = {INPUT_FASTA.resolve()}
    if LABELS_TSV is not None:
        input_paths.add(LABELS_TSV.resolve())
    if OUTPUT_FILE.resolve() in input_paths or METADATA_FILE.resolve() in input_paths:
        raise ValueError("输出路径不能与输入FASTA或标签文件相同")

    sample_ids, original_sequences = read_fasta(INPUT_FASTA)
    labels = read_labels(LABELS_TSV, sample_ids) if LABELS_TSV is not None else None
    features, known_mask, sequences, placeholder_type, placeholder_counts = encode_blosum62(original_sequences)
    if not np.isfinite(features).all():
        raise RuntimeError("BLOSUM62特征出现NaN或Inf，停止保存")

    arrays = {
        "X": features,
        "ids": np.asarray(sample_ids, dtype=str),
        "sequences": np.asarray(sequences, dtype=str),
        "original_sequences": np.asarray(original_sequences, dtype=str),
        "known_residue_mask": known_mask,
        "placeholder_mask": known_mask == 0,
        "x_placeholder_mask": placeholder_type == 1,
        "gap_placeholder_mask": placeholder_type == 2,
        "placeholder_type": placeholder_type,  # 0=canonical, 1=X, 2=-
        "placeholder_type_labels": np.asarray(["canonical", "X", "-"], dtype=str),
        "known_residue_count": known_mask.sum(axis=1, dtype=np.int16),
        "known_residue_fraction": known_mask.mean(axis=1, dtype=np.float32),
        "feature_columns": np.asarray(list(AMINO_ACIDS), dtype=str),
        "encoding_matrix": ENCODING_MATRIX,
    }
    if labels is not None:
        arrays["y"] = labels
        arrays["is_labeled_positive"] = labels
        arrays["label_semantics"] = np.asarray(LABEL_SEMANTICS)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(OUTPUT_FILE, **arrays)

    metadata = {
        "encoding": "BLOSUM62",
        "input_fasta": str(INPUT_FASTA.resolve()),
        "input_fasta_sha256": file_sha256(INPUT_FASTA),
        "labels_tsv": str(LABELS_TSV.resolve()) if LABELS_TSV is not None else None,
        "labels_tsv_sha256": file_sha256(LABELS_TSV) if LABELS_TSV is not None else None,
        "output_file": str(OUTPUT_FILE.resolve()),
        "sample_count": len(sample_ids),
        "feature_shape": list(features.shape),
        "fragment_length": FRAGMENT_LENGTH,
        "feature_columns": list(AMINO_ACIDS),
        "feature_dimension_per_residue": len(AMINO_ACIDS),
        "dtype": "float32",
        "normalization": "none; raw BLOSUM62 scores",
        "unknown_residue_policy": {
            "allowed_placeholders": sorted(PLACEHOLDER_SYMBOLS),
            "policy": PLACEHOLDER_POLICY,
            "encoding": "all-zero BLOSUM62 vector; no BLOSUM62 X score row is used",
            "placeholder_type": "0=canonical, 1=X, 2=-",
            "sequences_field": "X and - both normalized to X for backward compatibility; use original_sequences and placeholder_type to distinguish them",
        },
        "placeholder_counts": dict(placeholder_counts),
        "mask_definition": "1=standard amino acid; 0=X or - placeholder",
        "labels_included": labels is not None,
        "label_semantics": LABEL_SEMANTICS if labels is not None else None,
        "positive_count": int(labels.sum()) if labels is not None else None,
        "unlabeled_count": int(len(labels) - labels.sum()) if labels is not None else None,
        "input_order_preserved": True,
        "complete": True,
    }
    with METADATA_FILE.open("w" if ALLOW_OVERWRITE else "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"编码完成：X={features.shape}，占位符X/-总数={sum(placeholder_counts.values())}，"
          f"含占位符样本={int((known_mask == 0).any(axis=1).sum())}")
    if labels is not None:
        print(f"标签：P={int(labels.sum())}，U={int(len(labels) - labels.sum())}")
    else:
        print("未提供LABELS_TSV：NPZ中未保存y，程序没有猜测标签。")
    print(f"特征文件：{OUTPUT_FILE.resolve()}")
    print(f"元数据：{METADATA_FILE.resolve()}")
    print("融合其他编码前，务必按ids核对并对齐样本。")


if __name__ == "__main__":
    main()
