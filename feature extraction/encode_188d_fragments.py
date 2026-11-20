#!/usr/bin/env python3
"""对单个氨基酸片段 FASTA 进行 188D（CTD）编码，可在 IDE 中直接运行。

188D = 20 个氨基酸组成特征 + 8 个理化性质的 CTD 特征。
每项 CTD 特征包含 3 个 Composition、3 个 Transition、15 个 Distribution，
因此特征数为 20 + 8 * (3 + 3 + 15) = 188。

输入及输出约定与本项目其他编码器一致：每次处理一个 FASTA，按 sample_id
对齐可选标签表，保存 NPZ（X、ids、protein_ids、y 等）及 JSON 元数据。
188D 是整个 21 aa 片段的统计描述，所以 X 的形状是 [N, 188]，而不是逐残基
张量 [N, 21, D]。不要把 X 直接送入只接受逐残基 AAindex 输入的训练脚本。

X 和 - 可作为未知/缺失位置：不分配至任何 CTD 类别，且不计入组成和转移
的有效分母；Distribution 仍保持其在原片段中的位置。相关质量信息会额外
保存，不会改变 188D 的 188 个特征维度。

依赖：numpy。使用方法：在“IDE 运行配置区”修改路径后直接运行。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


# ============================== IDE 运行配置区 ==============================
INPUT_FASTA = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test.fasta"
)

# 混合P/U文件可配置配套标签表，表中必须有sample_id和label两列。
# 如果没有标签表或只需要编码，请设置为None；程序不会根据文件名猜标签。
LABELS_TSV: Path | None = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_label.tsv"
)

OUTPUT_FILE = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_188d.npz"
)
METADATA_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_metadata.json")

EXPECTED_FRAGMENT_LENGTH = 21
CENTER_POSITION = 11                 # 从 1 开始计数；188D 本身不使用中心位置
REQUIRE_PROTEIN_ID_FORMAT = True     # sample_id 应类似 EPB41_HUMAN_491
OVERWRITE_OUTPUTS = False            # 默认拒绝覆盖，避免误覆盖已有编码

# 188D 原始定义只含 20 种标准氨基酸。此处允许的 X 和 - 表示未知/缺失位置，
# 不会被伪装成任何理化类别。exclude 为推荐设置；error 用于严格复现仅含标准
# 氨基酸的旧 188D.jar 输入约束。B、Z、U、O、J 等仍视为非法，而不是猜测替换。
PLACEHOLDER_SYMBOLS = frozenset({"X", "-"})
PLACEHOLDER_POLICY = "exclude"       # exclude 或 error

# 仅在需要复现旧 188D.jar / 188.jar 结果时设置。参考 CSV 每行应为 188 个数值，
# 或为 188 个数值后再加一个 class 标签。校验只读，不会修改参考文件。
REFERENCE_188D_CSV: Path | None = None
REFERENCE_ATOL = 1e-5
# ============================================================================


# 说明文件要求的字母顺序；组成特征严格按该顺序输出。
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
LABEL_SEMANTICS_IF_STATUS_ABSENT = "label is read verbatim from labels TSV; consult its status column"

# 原 188D 工具使用三个类别 P/N/H。P 为低组、N 为中性组、H 为高组。
# Composition、Transition、Distribution 的组顺序分别为 P/N/H、P-N/P-H/N-H、
# P/N/H。Distribution 中的小分组保留旧工具的 0 值规则，详见函数注释。
PROPERTIES = (
    ("hydrophobicity", {
        "P": "RKEDQN", "N": "GASTPHY", "H": "CVLIMFW",
    }),
    ("van_der_waals_volume", {
        "P": "GASCTPD", "N": "NVEQIL", "H": "MHKFRYW",
    }),
    ("polarity", {
        "P": "LIFWCMVY", "N": "PATGS", "H": "HQRKNED",
    }),
    ("polarizability", {
        "P": "GASDT", "N": "CPNVEQIL", "H": "KMHFRYW",
    }),
    ("charge", {
        "P": "KR", "N": "ANCQGHILMFPSTWYV", "H": "DE",
    }),
    ("surface_tension", {
        "P": "GQDNAHR", "N": "KTSEC", "H": "ILMFPWYV",
    }),
    ("secondary_structure_propensity", {
        "P": "EALMQKRH", "N": "VIYCWFT", "H": "GNPSD",
    }),
    ("solvent_accessibility_propensity", {
        "P": "ALFCGIVW", "N": "RKQEND", "H": "MPSTHY",
    }),
)


def validate_property_groups() -> None:
    """防止人为编辑分组时引入遗漏或类别重叠。"""
    canonical = set(AA_ORDER)
    for property_name, groups in PROPERTIES:
        if set(groups) != {"P", "N", "H"}:
            raise RuntimeError(f"{property_name}: 分组键必须恰好为 P/N/H")
        sets = {key: set(value) for key, value in groups.items()}
        union = set().union(*sets.values())
        if union != canonical or sum(len(value) for value in sets.values()) != 20:
            raise RuntimeError(f"{property_name}: 三组必须不重不漏地覆盖 20 种标准氨基酸")


def make_feature_names() -> list[str]:
    names = [f"aa_composition_{aa}" for aa in AA_ORDER]
    percentile_names = ("first", "p25", "p50", "p75", "last")
    for property_name, _ in PROPERTIES:
        names.extend(f"{property_name}_composition_{group}" for group in ("P", "N", "H"))
        names.extend(f"{property_name}_transition_{left}_{right}"
                     for left, right in (("P", "N"), ("P", "H"), ("N", "H")))
        names.extend(f"{property_name}_distribution_{group}_{percentile}"
                     for group in ("P", "N", "H") for percentile in percentile_names)
    if len(names) != 188 or len(set(names)) != 188:
        raise RuntimeError("188D 特征名称数量或唯一性错误")
    return names


FEATURE_NAMES = make_feature_names()


def validate_placeholder_configuration() -> None:
    if PLACEHOLDER_POLICY not in {"exclude", "error"}:
        raise ValueError("PLACEHOLDER_POLICY 只能是 'exclude' 或 'error'")
    if not PLACEHOLDER_SYMBOLS or set(PLACEHOLDER_SYMBOLS) - {"X", "-"}:
        raise ValueError("PLACEHOLDER_SYMBOLS 只能包含 X 和 -，且不能为空")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """读取多行 FASTA；取标题第一个空白分隔字段为 sample_id。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 FASTA：{path}")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    identifier: str | None = None
    parts: list[str] = []

    def append_record() -> None:
        nonlocal identifier, parts
        if identifier is None:
            return
        sequence = "".join(parts).upper()
        if not sequence:
            raise ValueError(f"{path}: {identifier} 的序列为空")
        if identifier in seen:
            raise ValueError(f"{path}: 存在重复 sample_id：{identifier}")
        seen.add(identifier)
        records.append((identifier, sequence))

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                title = line[1:].strip()
                if not title:
                    raise ValueError(f"{path}:{line_number}: FASTA 标题为空")
                identifier, parts = title.split()[0], []
            else:
                if identifier is None:
                    raise ValueError(f"{path}:{line_number}: 第一个 FASTA 标题之前出现序列")
                parts.append("".join(line.split()))
    append_record()
    if not records:
        raise ValueError(f"{path}: FASTA 中没有记录")
    return records


def protein_id(sample_id: str) -> str:
    """EPB41_HUMAN_491 -> EPB41_HUMAN；不使用模糊的字符串替换。"""
    protein, separator, site = sample_id.rpartition("_")
    if protein and separator and site.isdigit():
        return protein
    if REQUIRE_PROTEIN_ID_FORMAT:
        raise ValueError(
            f"sample_id={sample_id!r} 不符合 蛋白质名_位点 格式；"
            "请修正输入或将 REQUIRE_PROTEIN_ID_FORMAT 设为 False。"
        )
    return ""


def read_labels(path: Path, fasta_ids: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """按 sample_id 而非文件行号连接标签；status 是可选列。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到标签 TSV：{path}")
    mapping: dict[str, int] = {}
    statuses: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"sample_id", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} 必须包含 sample_id 和 label 两列")
        has_status = "status" in reader.fieldnames
        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not sample_id or sample_id in mapping:
                raise ValueError(f"{path}:{line_number}: sample_id 为空或重复：{sample_id!r}")
            if label_text not in {"0", "1"}:
                raise ValueError(f"{path}:{line_number}: label 必须是 0 或 1")
            mapping[sample_id] = int(label_text)
            if has_status:
                statuses[sample_id] = (row.get("status") or "").strip()

    fasta_set = set(fasta_ids)
    missing = [item for item in fasta_ids if item not in mapping]
    extra = sorted(set(mapping) - fasta_set)
    if missing or extra:
        raise ValueError(
            f"FASTA 与标签表 ID 不完全一致：缺失={len(missing)}，多出={len(extra)}；"
            f"示例 missing={missing[:3]}，extra={extra[:3]}"
        )
    y = np.asarray([mapping[item] for item in fasta_ids], dtype=np.uint8)
    status_array = (np.asarray([statuses[item] for item in fasta_ids], dtype=str)
                    if statuses else None)
    return y, status_array


def prepare_input() -> tuple[list[str], list[str], list[str], np.ndarray | None, np.ndarray | None]:
    if EXPECTED_FRAGMENT_LENGTH < 1 or not 1 <= CENTER_POSITION <= EXPECTED_FRAGMENT_LENGTH:
        raise ValueError("EXPECTED_FRAGMENT_LENGTH 或 CENTER_POSITION 配置无效")
    ids, sequences, proteins = [], [], []
    for sample_id, sequence in read_fasta(INPUT_FASTA):
        if len(sequence) != EXPECTED_FRAGMENT_LENGTH:
            raise ValueError(
                f"{sample_id}: 长度为 {len(sequence)}，预期 {EXPECTED_FRAGMENT_LENGTH}；"
                "程序不会自动截断或填充。"
            )
        invalid = sorted(set(sequence) - set(AA_ORDER) - set(PLACEHOLDER_SYMBOLS))
        placeholders = sorted(set(sequence) & set(PLACEHOLDER_SYMBOLS))
        if invalid:
            raise ValueError(
                f"{sample_id}: 发现不支持的残基 {invalid}。仅允许 20 种标准氨基酸、"
                f"以及占位符 {sorted(PLACEHOLDER_SYMBOLS)}。"
            )
        if placeholders and PLACEHOLDER_POLICY == "error":
            raise ValueError(
                f"{sample_id}: 发现占位符 {placeholders}；当前 PLACEHOLDER_POLICY='error'。"
            )
        ids.append(sample_id)
        sequences.append(sequence)
        proteins.append(protein_id(sample_id))
    y, status = read_labels(LABELS_TSV, ids) if LABELS_TSV is not None else (None, None)
    return ids, sequences, proteins, y, status


def distribution_positions(group_positions: list[int], length: int) -> list[float]:
    """输出 first, 25%, 50%, 75%, last，单位为序列百分比。

    与 188D.jar 历史 CSV 相同：25/50/75% 使用 floor(n * q) 选择第 k 个出现位置。
    当 k=0（例如该组仅出现 1--3 次时的 25%）时，该特征保留为 0，而不使用
    第一个出现位置填补。组内没有残基时，5 个特征均为 0。
    """
    if not group_positions:
        return [0.0] * 5
    count = len(group_positions)
    ranks = (1, int(np.floor(count * 0.25)), int(np.floor(count * 0.50)),
             int(np.floor(count * 0.75)), count)
    return [100.0 * group_positions[rank - 1] / length if rank else 0.0 for rank in ranks]


def known_residue_mask(sequence: str) -> np.ndarray:
    """返回逐位置标准氨基酸掩码；X/- 为 False。"""
    return np.asarray([amino_acid in AA_ORDER for amino_acid in sequence], dtype=bool)


def encode_sequence(sequence: str) -> np.ndarray:
    """按 ReadMe-python.txt 的分组生成 188D；X/- 按配置作为缺失位置处理。"""
    sequence = sequence.upper()
    length = len(sequence)
    invalid = set(sequence) - set(AA_ORDER) - set(PLACEHOLDER_SYMBOLS)
    if length == 0 or invalid:
        raise ValueError("encode_sequence 只接受非空序列，且残基仅可为标准氨基酸、X 或 -")
    known = known_residue_mask(sequence)
    if not known.all() and PLACEHOLDER_POLICY == "error":
        raise ValueError("发现 X/-，但 PLACEHOLDER_POLICY='error'")
    known_count = int(known.sum())

    # Composition 的分母为已知残基数，故已知残基存在时前 20 维总和始终为 100。
    values: list[float] = [100.0 * sequence.count(amino_acid) / known_count if known_count else 0.0
                           for amino_acid in AA_ORDER]
    for _, groups in PROPERTIES:
        category = {amino_acid: group for group, amino_acids in groups.items()
                    for amino_acid in amino_acids}
        states = [category.get(amino_acid) for amino_acid in sequence]

        # Composition：P/N/H，和说明文件及历史 188D CSV 保持一致。
        values.extend(100.0 * states.count(group) / known_count if known_count else 0.0
                      for group in ("P", "N", "H"))

        # 仅计算两端均为标准氨基酸的相邻对；X/- 既不制造也不连接一次转移。
        valid_pairs = [index for index in range(length - 1)
                       if states[index] is not None and states[index + 1] is not None]
        denominator = len(valid_pairs)
        for left, right in (("P", "N"), ("P", "H"), ("N", "H")):
            changes = sum(
                {states[index], states[index + 1]} == {left, right}
                for index in valid_pairs
            )
            values.append(100.0 * changes / denominator if denominator else 0.0)

        # Distribution：位置使用 1 开始编号，随后除以 L 并乘 100。
        for group in ("P", "N", "H"):
            positions = [index + 1 for index, state in enumerate(states) if state == group]
            values.extend(distribution_positions(positions, length))

    result = np.asarray(values, dtype=np.float32)
    if result.shape != (188,) or not np.isfinite(result).all():
        raise RuntimeError(f"188D 特征形状或数值无效：{result.shape}")
    return result


def verify_reference(encoded: np.ndarray) -> None:
    """可选的历史 CSV 校验：用于确认本实现与旧 jar 完全一致。"""
    if REFERENCE_188D_CSV is None:
        return
    if not REFERENCE_188D_CSV.is_file():
        raise FileNotFoundError(f"找不到参考 CSV：{REFERENCE_188D_CSV}")
    rows: list[list[float]] = []
    with REFERENCE_188D_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            numeric = row[:188]
            if len(numeric) != 188:
                raise ValueError(f"参考 CSV 第 {line_number} 行不足 188 个特征")
            try:
                rows.append([float(item) for item in numeric])
            except ValueError as exc:
                raise ValueError(f"参考 CSV 第 {line_number} 行不是数值特征") from exc
    reference = np.asarray(rows, dtype=np.float64)
    if reference.shape != encoded.shape:
        raise ValueError(f"参考 CSV 形状 {reference.shape} 与本次编码 {encoded.shape} 不一致")
    maximum_error = float(np.max(np.abs(reference - encoded.astype(np.float64))))
    if not np.allclose(reference, encoded, rtol=0.0, atol=REFERENCE_ATOL):
        index = np.unravel_index(np.argmax(np.abs(reference - encoded)), encoded.shape)
        raise ValueError(
            f"与参考 CSV 不一致：最大绝对误差={maximum_error:.8g}，"
            f"样本行={index[0]}，特征={FEATURE_NAMES[index[1]]}。"
        )
    print(f"已通过参考 CSV 校验：{REFERENCE_188D_CSV}；最大绝对误差={maximum_error:.3g}")


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists() and not OVERWRITE_OUTPUTS:
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


def main() -> None:
    validate_property_groups()
    validate_placeholder_configuration()
    ids, sequences, proteins, labels, status = prepare_input()
    output_targets = (OUTPUT_FILE, METADATA_FILE)
    exists = [str(path) for path in output_targets if path.exists()]
    if exists and not OVERWRITE_OUTPUTS:
        raise FileExistsError("以下输出已存在；请修改输出路径或显式设置 OVERWRITE_OUTPUTS=True：\n" + "\n".join(exists))
    inputs = {INPUT_FASTA.resolve()}
    if LABELS_TSV is not None:
        inputs.add(LABELS_TSV.resolve())
    if REFERENCE_188D_CSV is not None:
        inputs.add(REFERENCE_188D_CSV.resolve())
    if any(path.resolve() in inputs for path in output_targets):
        raise ValueError("输出路径不能覆盖输入 FASTA、标签表或参考 CSV")

    encoded = np.vstack([encode_sequence(sequence) for sequence in sequences]).astype(np.float32, copy=False)
    known_mask = np.vstack([known_residue_mask(sequence) for sequence in sequences])
    known_count = known_mask.sum(axis=1, dtype=np.int16)
    valid_transition_count = (known_mask[:, :-1] & known_mask[:, 1:]).sum(axis=1, dtype=np.int16)
    verify_reference(encoded)
    arrays: dict[str, np.ndarray] = {
        "X": encoded,
        "ids": np.asarray(ids, dtype=str),
        "protein_ids": np.asarray(proteins, dtype=str),
        "sequences": np.asarray(sequences, dtype=str),
        "raw_sequences": np.asarray(sequences, dtype=str),
        "known_residue_mask": known_mask,
        "known_residue_count": known_count,
        "known_residue_fraction": known_count.astype(np.float32) / EXPECTED_FRAGMENT_LENGTH,
        "valid_adjacent_transition_count": valid_transition_count,
        "feature_names": np.asarray(FEATURE_NAMES, dtype=str),
        # index_ids 作为通用别名，便于与部分现有特征检查工具兼容。
        "index_ids": np.asarray(FEATURE_NAMES, dtype=str),
        "encoding": np.asarray("188D_CTD"),
        "feature_count": np.asarray(188),
        "feature_scale": np.asarray("percentage_0_to_100"),
        "center_position_1_based": np.asarray(CENTER_POSITION),
        "fragment_length": np.asarray(EXPECTED_FRAGMENT_LENGTH),
    }
    if labels is not None:
        arrays["y"] = labels
        arrays["is_labeled_positive"] = labels
        if status is not None:
            arrays["label_status"] = status
    atomic_save_npz(OUTPUT_FILE, **arrays)

    placeholders = Counter(amino_acid for sequence in sequences for amino_acid in sequence
                           if amino_acid in PLACEHOLDER_SYMBOLS)
    metadata = {
        "encoding": "188D_CTD",
        "description": "20 amino-acid composition + 8 CTD property groups (3C+3T+15D each)",
        "feature_count": 188,
        "feature_names": FEATURE_NAMES,
        "feature_scale": "percentage_0_to_100; raw values; no dataset-fitted scaling",
        "placeholder_policy": {
            "symbols": sorted(PLACEHOLDER_SYMBOLS),
            "policy": PLACEHOLDER_POLICY,
            "composition_denominator": "number of canonical amino acids; all-zero if no canonical amino acid",
            "transition_denominator": "number of adjacent pairs whose two residues are canonical; all-zero if none",
            "distribution_positions": "canonical residues retain their original 1-based positions; divisor is full fragment length",
            "quality_arrays": ["known_residue_mask", "known_residue_count", "known_residue_fraction", "valid_adjacent_transition_count"],
        },
        "ctd_compatibility": {
            "composition_order": ["P", "N", "H"],
            "transition_order": ["P-N", "P-H", "N-H"],
            "distribution_order": ["P:first,p25,p50,p75,last", "N:first,p25,p50,p75,last", "H:first,p25,p50,p75,last"],
            "distribution_percentile_rule": "floor(group_count * q); rank 0 is encoded as 0; other positions are 1-based, then / sequence_length * 100",
        },
        "properties": [{"name": name, "groups": groups} for name, groups in PROPERTIES],
        "input_fasta": str(INPUT_FASTA.resolve()),
        "input_fasta_sha256": hashlib.sha256(INPUT_FASTA.read_bytes()).hexdigest(),
        "labels_tsv": str(LABELS_TSV.resolve()) if LABELS_TSV is not None else None,
        "labels_tsv_sha256": (hashlib.sha256(LABELS_TSV.read_bytes()).hexdigest()
                              if LABELS_TSV is not None else None),
        "labels_included": labels is not None,
        "label_status_included": status is not None,
        "label_status_counts": dict(Counter(status.tolist())) if status is not None else None,
        "label_semantics": ("preserved from labels TSV; inspect label_status" if status is not None
                            else LABEL_SEMANTICS_IF_STATUS_ABSENT if labels is not None else None),
        "axis_order": ["sample", "188D_feature"],
        "shape": list(encoded.shape),
        "dtype": "float32",
        "sample_count": len(ids),
        "positive_count": int(labels.sum()) if labels is not None else None,
        "zero_label_count": int(len(labels) - labels.sum()) if labels is not None else None,
        "fragment_length": EXPECTED_FRAGMENT_LENGTH,
        "center_position_1_based": CENTER_POSITION,
        "placeholder_counts": dict(placeholders),
        "reference_csv": str(REFERENCE_188D_CSV.resolve()) if REFERENCE_188D_CSV else None,
        "output_file": str(OUTPUT_FILE.resolve()),
        "input_order_preserved": True,
        "complete": True,
    }
    with METADATA_FILE.open("w" if OVERWRITE_OUTPUTS else "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"完成 188D 编码：X={encoded.shape}，特征范围=[{encoded.min():.4g}, {encoded.max():.4g}]")
    if labels is None:
        print("未提供标签表：NPZ 中不含 y，程序没有猜测标签。")
    else:
        print(f"标签已按 sample_id 对齐：label=1 为 {int(labels.sum())} 条，label=0 为 {len(labels) - int(labels.sum())} 条。")
    print(f"特征文件：{OUTPUT_FILE.resolve()}")
    print(f"元数据：{METADATA_FILE.resolve()}")
    print("训练时如需标准化，只能在训练集拟合 188 个特征的均值/标准差，再应用到验证集和测试集。")


if __name__ == "__main__":
    main()
