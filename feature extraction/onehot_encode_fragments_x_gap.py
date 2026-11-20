#!/usr/bin/env python3
"""对单个FASTA文件进行One-hot编码并保存为NPZ文件。

在IDE中打开本文件，按需修改“IDE运行配置区”，然后直接点击运行。
不需要提供命令行参数。

支持 X 和 - 占位符：默认保持原有 21 通道，将 - 映射到 X 通道；也可切换为
22 通道并显式区分 X 与 -。两种方式都会保存占位符位置和类型。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


# ========================== IDE运行配置区 ==========================
# 每次只编码一个FASTA；依次处理02～05时，同步修改下面三个文件名。
INPUT_FASTA = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test.fasta"
)

# 混合P/U文件可配置配套标签表，表中必须有sample_id和label两列。
# 如果没有标签表或只需要编码，请设置为None；程序不会根据文件名猜标签。
LABELS_TSV: Path | None = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_label.tsv"
)

OUTPUT_FILE = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_onehot.npz"
)
METADATA_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_metadata.json")

# 当前数据片段长度为21，候选修饰位点位于第11位。
EXPECTED_FRAGMENT_LENGTH = 21
CENTER_POSITION = 11  # 从1开始计数

ALLOW_OVERWRITE = False

# shared_X_channel：保持原有 20 个标准残基 + X 共 21 通道；- 编码到 X 通道。

PLACEHOLDER_ENCODING = "shared_X_channel"
# ==================================================================


# 仅允许 20 种标准氨基酸和两个明确的占位符。B/Z/U/O/J 没有唯一含义，拒绝而不猜测。
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
UNKNOWN_TOKEN = "X"
GAP_TOKEN = "-"
PLACEHOLDER_SYMBOLS = frozenset({UNKNOWN_TOKEN, GAP_TOKEN})


def active_alphabet() -> str:
    if PLACEHOLDER_ENCODING == "shared_X_channel":
        return STANDARD_AMINO_ACIDS + UNKNOWN_TOKEN
    if PLACEHOLDER_ENCODING == "separate_X_gap_channels":
        return STANDARD_AMINO_ACIDS + UNKNOWN_TOKEN + GAP_TOKEN
    raise ValueError(
        "PLACEHOLDER_ENCODING 只能是 'shared_X_channel' 或 'separate_X_gap_channels'"
    )


ALPHABET = active_alphabet()
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(ALPHABET)}
LABEL_SEMANTICS = "1=labeled_positive(P); 0=unlabeled(U), NOT verified_negative"


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """读取支持多行序列的FASTA文件，返回样本ID和序列。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    sample_ids: list[str] = []
    sequences: list[str] = []
    header: str | None = None
    sequence_parts: list[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    sample_ids.append(header)
                    sequences.append("".join(sequence_parts).upper())

                header = line[1:].strip().split(maxsplit=1)[0]
                if not header:
                    raise ValueError(f"{path} 第{line_number}行的FASTA标题为空")
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(
                        f"{path} 第{line_number}行在第一个FASTA标题之前出现序列"
                    )
                sequence_parts.append("".join(line.split()))

    if header is not None:
        sample_ids.append(header)
        sequences.append("".join(sequence_parts).upper())

    if not sample_ids:
        raise ValueError(f"{path} 中没有找到FASTA记录")
    if any(not sequence for sequence in sequences):
        raise ValueError(f"{path} 中至少存在一条空序列")

    duplicate_ids = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"{path} 中存在重复样本ID：{duplicate_ids[:5]}")

    return sample_ids, sequences


def normalize_sequence(sequence: str) -> tuple[str, np.ndarray, Counter[str]]:
    """规范化一个序列并保留占位符类型。

    返回值中的 placeholder_type：0=标准氨基酸，1=X，2=-。shared_X_channel 模式下，
    - 在 sequences 字段中归一化为 X 以兼容既有 21 通道模型；原始字符始终保存在
    original_sequences 及 placeholder_type 中。
    """
    replacement_counts: Counter[str] = Counter()
    normalized: list[str] = []
    placeholder_type = np.zeros(len(sequence), dtype=np.uint8)
    for position, residue in enumerate(sequence.upper()):
        if residue in STANDARD_AMINO_ACIDS:
            normalized.append(residue)
        elif residue == UNKNOWN_TOKEN:
            normalized.append(UNKNOWN_TOKEN)
            placeholder_type[position] = 1
            replacement_counts[residue] += 1
        elif residue == GAP_TOKEN:
            normalized.append(GAP_TOKEN if PLACEHOLDER_ENCODING == "separate_X_gap_channels"
                              else UNKNOWN_TOKEN)
            placeholder_type[position] = 2
            replacement_counts[residue] += 1
        else:
            raise ValueError(
                f"发现不支持的残基 {residue!r}；仅允许20种标准氨基酸、X或-，"
                "不对 B/Z/U/O/J 做猜测替换。"
            )
    return "".join(normalized), placeholder_type, replacement_counts


def encode_sequences(
    sequences: list[str], expected_length: int
) -> tuple[np.ndarray, list[str], np.ndarray, Counter[str], str]:
    """将等长序列编码为 (N,L,21) 或 (N,L,22) 的 uint8 One-hot 数组。"""
    alphabet = active_alphabet()
    aa_to_index = {amino_acid: index for index, amino_acid in enumerate(alphabet)}
    invalid_lengths = [
        (index, len(sequence))
        for index, sequence in enumerate(sequences)
        if len(sequence) != expected_length
    ]
    if invalid_lengths:
        examples = ", ".join(
            f"样本索引{index}:长度{length}"
            for index, length in invalid_lengths[:5]
        )
        raise ValueError(
            f"发现{len(invalid_lengths)}条长度不等于{expected_length}的序列；{examples}"
        )

    encoded = np.zeros((len(sequences), expected_length, len(alphabet)), dtype=np.uint8)
    placeholder_types = np.zeros((len(sequences), expected_length), dtype=np.uint8)
    normalized_sequences: list[str] = []
    placeholder_counts: Counter[str] = Counter()

    for sample_index, sequence in enumerate(sequences):
        normalized, types, replacements = normalize_sequence(sequence)
        normalized_sequences.append(normalized)
        placeholder_types[sample_index] = types
        placeholder_counts.update(replacements)
        residue_indices = np.fromiter(
            (aa_to_index[residue] for residue in normalized),
            dtype=np.int64,
            count=expected_length,
        )
        encoded[sample_index, np.arange(expected_length), residue_indices] = 1

    # 即使存在 X/-，每个位置也必须恰有一个有效通道。
    if not np.all(encoded.sum(axis=2) == 1):
        raise RuntimeError("内部编码错误：存在未编码或多重编码的残基位置")
    return encoded, normalized_sequences, placeholder_types, placeholder_counts, alphabet


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays) -> None:
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
    if not 1 <= CENTER_POSITION <= EXPECTED_FRAGMENT_LENGTH:
        raise ValueError("CENTER_POSITION必须位于片段范围内")

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

    sample_ids, sequences = read_fasta(INPUT_FASTA)
    labels = read_labels(LABELS_TSV, sample_ids) if LABELS_TSV is not None else None
    features, normalized_sequences, placeholder_type, replacements, alphabet = encode_sequences(
        sequences, EXPECTED_FRAGMENT_LENGTH
    )
    known_mask = placeholder_type == 0

    arrays = {
        "X": features,
        "ids": np.asarray(sample_ids, dtype=str),
        "sequences": np.asarray(normalized_sequences, dtype=str),
        "original_sequences": np.asarray(sequences, dtype=str),
        "known_residue_mask": known_mask,
        "placeholder_mask": ~known_mask,
        "x_placeholder_mask": placeholder_type == 1,
        "gap_placeholder_mask": placeholder_type == 2,
        "placeholder_type": placeholder_type,  # 0=canonical, 1=X, 2=-
        "placeholder_type_labels": np.asarray(["canonical", "X", "-"], dtype=str),
        "known_residue_count": known_mask.sum(axis=1, dtype=np.int16),
        "known_residue_fraction": known_mask.mean(axis=1, dtype=np.float32),
        "alphabet": np.asarray(list(alphabet), dtype=str),
    }
    if labels is not None:
        arrays["y"] = labels
        arrays["is_labeled_positive"] = labels
        arrays["label_semantics"] = np.asarray(LABEL_SEMANTICS)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(OUTPUT_FILE, **arrays)

    metadata = {
        "encoding": "one_hot",
        "input_fasta": str(INPUT_FASTA.resolve()),
        "input_fasta_sha256": file_sha256(INPUT_FASTA),
        "labels_tsv": str(LABELS_TSV.resolve()) if LABELS_TSV is not None else None,
        "labels_tsv_sha256": file_sha256(LABELS_TSV) if LABELS_TSV is not None else None,
        "output_file": str(OUTPUT_FILE.resolve()),
        "sample_count": len(sample_ids),
        "feature_shape": list(features.shape),
        "alphabet": alphabet,
        "channels": len(alphabet),
        "unknown_token": UNKNOWN_TOKEN,
        "placeholder_policy": {
            "mode": PLACEHOLDER_ENCODING,
            "allowed_placeholders": sorted(PLACEHOLDER_SYMBOLS),
            "shared_X_channel": "X and - both use X channel; 21 channels total",
            "separate_X_gap_channels": "X and - use separate channels; 22 channels total",
            "placeholder_type": "0=canonical, 1=X, 2=-",
        },
        "placeholder_counts": dict(replacements),
        "fragment_length": EXPECTED_FRAGMENT_LENGTH,
        "center_position_1_based": CENTER_POSITION,
        "array_dtype": "uint8",
        "labels_included": labels is not None,
        "label_semantics": LABEL_SEMANTICS if labels is not None else None,
        "positive_count": int(labels.sum()) if labels is not None else None,
        "unlabeled_count": int(len(labels) - labels.sum()) if labels is not None else None,
        "input_order_preserved": True,
        "complete": True,
    }
    with METADATA_FILE.open("w" if ALLOW_OVERWRITE else "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"编码完成：X={features.shape}")
    if labels is not None:
        print(f"标签：P={int(labels.sum())}，U={int(len(labels) - labels.sum())}")
    else:
        print("未提供LABELS_TSV：NPZ中未保存y，程序没有猜测标签。")
    if replacements:
        details = ", ".join(f"{character}: {count}个" for character, count in sorted(replacements.items()))
        print(f"占位符处理模式={PLACEHOLDER_ENCODING}；{details}")
    else:
        print("所有残基均为20种标准氨基酸，无占位符。")
    print(f"特征文件：{OUTPUT_FILE.resolve()}")
    print(f"元数据：{METADATA_FILE.resolve()}")
    print("融合其他编码前，务必按ids核对并对齐样本。")


if __name__ == "__main__":
    main()
