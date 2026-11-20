#!/usr/bin/env python3
"""对单个FASTA文件进行AAindex1逐残基编码，可在IDE中直接运行。

依赖：numpy（在IDE所用Python环境执行 python -m pip install numpy）。
首次自动下载GenomeNet的AAindex1数据库；也可以手动下载至AAINDEX_FILE。
每次仅处理INPUT_FASTA指定的一个文件；输出X形状为(N, L, K)。
LABELS_TSV可选：提供时保存y；不提供时不猜测标签，也不保存y。
AAindex是残基属性查表编码，不是该片段真实三维结构的预测。
X 和 - 可作为未知/缺失占位符：二者均以每个 AAindex 指标的 20 种氨基酸均值编码
（STANDARDIZE=True 时为 0），并额外保存不同的占位符掩码，避免混淆其来源。
"""

from __future__ import annotations

import hashlib
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("请在IDE使用的Python环境安装依赖：python -m pip install numpy") from exc


# ======================== IDE运行配置区 ========================
INPUT_FASTA = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test.fasta"
)

# 混合P/U文件可配置配套标签表，表中必须有sample_id和label两列。
# 如果没有标签表或只需要编码，请设置为None；程序不会根据文件名猜标签。
LABELS_TSV: Path | None = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_label.tsv"
)

OUTPUT_FILE = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_aaindex.npz"
)
# 元数据与NPZ放在同一目录。依次编码02～05时只需同步修改上面三个文件名。
METADATA_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_metadata.json")

EXPECTED_FRAGMENT_LENGTH = 21
CENTER_POSITION = 11                 # 从1开始计数，不要求中心一定为S/T
REQUIRE_PROTEIN_ID_FORMAT = True     # ID须为蛋白质名_位点，例如EPB41_HUMAN_491
OVERWRITE_OUTPUTS = False            # 防止覆盖旧特征；重跑时建议使用新输出目录

AAINDEX_URL = "https://www.genome.jp/ftp/db/community/aaindex/aaindex1"
AAINDEX_FILE = OUTPUT_FILE.parent / "aaindex1.txt"
AUTO_DOWNLOAD = True                # 离线时手动下载，并设置正确的AAINDEX_FILE
DOWNLOAD_TIMEOUT_SECONDS = 45

# 默认采用全部20个值均完整且方差非零的指标，按编号排序。
# 已验证的566条目数据库中，跳过13个含缺失值的指标后，保留553个。
# 实际维数以运行日志和保存的index_ids为准，不将553硬编码到模型中。
# 自定义时请使用AAindex1编号；AAindex2/3不适用于这里的逐残基查表。
INDEX_IDS = None

# 每个指标在20种标准氨基酸之间做z-score（ddof=0），不从样本估计均值。
# 因此训练/验证/测试使用完全同一查表，不会使用测试数据拟合预处理。
STANDARDIZE = True
# X 和 - 可作为未知/缺失占位符，均用该指标的20种氨基酸均值表示；
# 标准化后为 0。二者不会替换为任意标准氨基酸，并另存各自的掩码。
# B/Z/U/O/J 不具有唯一的 AAindex 定义，默认拒绝，避免不透明的猜测替换。
PLACEHOLDER_POLICY = "mean_with_mask"  # mean_with_mask 或 error
# ============================================================== 

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
PLACEHOLDER_SYMBOLS = frozenset({"X", "-"})
LABEL_SEMANTICS = "1=labeled_positive(P); 0=unlabeled(U), NOT verified_negative"


def parse_aaindex1(text: str) -> dict:
    """解析官方AAindex1格式，特别注意I行的A/L等是上下两行，而非交错排列。"""
    records = {}
    chunks = text.split("//")
    if chunks[-1].strip():
        raise ValueError("AAindex1缺少末尾//，文件可能不完整或不是官方文本格式")
    for block in chunks[:-1]:
        if not block.strip():
            continue
        accession = None
        description = []
        order = None
        numbers = []
        current_field = None
        for line in block.splitlines():
            if not line.strip():
                continue
            if not line[0].isspace():
                current_field = line[0]
                if current_field == "H":
                    accession = line[1:].strip()
                elif current_field == "D":
                    description.append(line[1:].strip())
                elif current_field == "I":
                    pairs = line[1:].split()
                    if len(pairs) != 10 or any(len(p) != 3 or p[1] != "/" for p in pairs):
                        raise ValueError(f"{accession}: I行氨基酸顺序格式不正确")
                    order = "".join(p[0] for p in pairs) + "".join(p[2] for p in pairs)
            elif current_field == "D":
                description.append(line.strip())
            elif current_field == "I":
                for token in line.split():
                    try:
                        numbers.append(float(token) if token.upper() not in {"NA", "N/A", "-"} else np.nan)
                    except ValueError as exc:
                        raise ValueError(f"{accession}: 非法数值{token!r}") from exc
        if not accession or order is None or len(numbers) != 20:
            raise ValueError(f"AAindex1条目不完整：{accession!r}，数值个数={len(numbers)}")
        if len(set(order)) != 20 or set(order) != set(AA_ORDER):
            raise ValueError(f"{accession}: 非法氨基酸顺序{order}")
        if accession in records:
            raise ValueError(f"AAindex1重复编号：{accession}")
        values = np.asarray(numbers, dtype=np.float64)
        records[accession] = {
            "description": " ".join(description),
            "values": values[[order.index(aa) for aa in AA_ORDER]],
        }
    if not records:
        raise ValueError("没有解析到AAindex1条目；请勿将网页HTML保存为数据库")
    return records


def read_database() -> tuple[dict, str]:
    if not AAINDEX_FILE.is_file():
        if not AUTO_DOWNLOAD:
            raise FileNotFoundError(f"找不到数据库，请手动下载：{AAINDEX_URL}\n保存至：{AAINDEX_FILE}")
        print(f"首次下载AAindex1：{AAINDEX_URL}", flush=True)
        try:
            request = Request(AAINDEX_URL, headers={"User-Agent": "AAindexFragmentEncoder/1.0"})
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                content = response.read(5_000_001)
            if len(content) > 5_000_000:
                raise ValueError("数据库文件大小异常，超过5 MB")
            parse_aaindex1(content.decode("utf-8-sig"))  # 验证后才缓存，避免保存下载错误页面
        except Exception as exc:
            raise RuntimeError(
                f"自动下载/验证失败：{exc}\n请用浏览器下载{AAINDEX_URL}\n"
                f"保存到{AAINDEX_FILE}后重跑；不要关闭HTTPS证书验证。"
            ) from exc
        AAINDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AAINDEX_FILE.open("xb") as handle:
            handle.write(content)
    content = AAINDEX_FILE.read_bytes()
    return parse_aaindex1(content.decode("utf-8-sig")), hashlib.sha256(content).hexdigest()


def build_lookup(records: dict, selected_ids, standardize: bool) -> dict:
    """返回20个标准残基+1个未知残基的查表；缺失指标不静默插补。"""
    ids = sorted(records) if selected_ids is None else list(selected_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("INDEX_IDS不能为空或包含重复编号")
    kept, raw_columns, skipped = [], [], {}
    for accession in ids:
        if accession not in records:
            raise ValueError(f"AAindex1中不存在编号：{accession}")
        values = records[accession]["values"]
        reason = None
        if not np.isfinite(values).all():
            reason = "missing_or_nonfinite_values"
        elif values.std(ddof=0) <= 1e-12:
            reason = "zero_variance"
        if reason:
            if selected_ids is not None:
                raise ValueError(f"所选指标{accession}不可用：{reason}。请更换指标。")
            skipped[accession] = reason
            continue
        kept.append(accession)
        raw_columns.append(values)
    if not kept:
        raise ValueError("没有完整且方差非零的AAindex指标")
    raw = np.column_stack(raw_columns)
    mean, std = raw.mean(axis=0), raw.std(axis=0, ddof=0)
    standard = (raw - mean) / std if standardize else raw
    unknown = np.zeros(len(kept)) if standardize else mean
    lookup = np.vstack([standard, unknown]).astype(np.float32)
    return dict(ids=kept, raw=raw, mean=mean, std=std, lookup=lookup,
                descriptions=[records[i]["description"] for i in kept], skipped=skipped)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """保留标题第一个字段作为ID；支持多行、空白、小写和UTF-8 BOM。"""
    result, seen = [], set()
    identifier, parts = None, []

    def append_record():
        if identifier is None:
            return
        sequence = "".join(parts).upper()
        if not sequence:
            raise ValueError(f"{path}: {identifier}为空序列")
        if identifier in seen:
            raise ValueError(f"{path}: 重复ID {identifier}")
        seen.add(identifier)
        result.append((identifier, sequence))

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                title = line[1:].strip()
                if not title:
                    raise ValueError(f"{path}:{line_number}: FASTA标题为空")
                identifier, parts = title.split()[0], []
            else:
                if identifier is None:
                    raise ValueError(f"{path}:{line_number}: 序列位于第一个标题前")
                parts.append("".join(line.split()))
    append_record()
    if not result:
        raise ValueError(f"没有FASTA记录：{path}")
    return result


def protein_id(identifier: str) -> str:
    protein, separator, site = identifier.rpartition("_")
    if protein and separator and site.isdigit():
        return protein
    if REQUIRE_PROTEIN_ID_FORMAT:
        raise ValueError(f"ID {identifier!r}不是蛋白质名_位点格式；请修改protein_id解析函数")
    return ""  # 不伪造蛋白质ID


def read_labels(path: Path, fasta_ids: list[str]) -> np.ndarray:
    """按sample_id对齐标签；TSV行顺序可以与FASTA不同。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到标签文件：{path}")
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
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

    fasta_set = set(fasta_ids)
    missing = [sample_id for sample_id in fasta_ids if sample_id not in mapping]
    extra = sorted(set(mapping) - fasta_set)
    if missing or extra:
        raise ValueError(
            f"FASTA与标签表ID不完全一致：缺少标签{len(missing)}个，"
            f"标签表多出{len(extra)}个；示例：missing={missing[:3]}, extra={extra[:3]}"
        )
    return np.asarray([mapping[sample_id] for sample_id in fasta_ids], dtype=np.uint8)


def load_input() -> tuple[list[str], list[str], np.ndarray | None, list[str]]:
    """读取并验证一个FASTA，以及可选的逐样本标签。"""
    if EXPECTED_FRAGMENT_LENGTH < 1 or not 1 <= CENTER_POSITION <= EXPECTED_FRAGMENT_LENGTH:
        raise ValueError("片段长度或中心位置配置错误")
    records = read_fasta(INPUT_FASTA)
    ids, sequences, proteins = [], [], []
    for identifier, sequence in records:
        if len(sequence) != EXPECTED_FRAGMENT_LENGTH:
            raise ValueError(
                f"{identifier}: 长度{len(sequence)}，预期{EXPECTED_FRAGMENT_LENGTH}；"
                "程序不会自动截断或补齐"
            )
        invalid = set(sequence) - set(AA_ORDER) - set(PLACEHOLDER_SYMBOLS)
        placeholders = set(sequence) & set(PLACEHOLDER_SYMBOLS)
        if invalid:
            raise ValueError(
                f"{identifier}: 非法序列字符{sorted(invalid)}；仅允许20种标准氨基酸以及"
                f"占位符{sorted(PLACEHOLDER_SYMBOLS)}。B/Z/U/O/J不能被自动猜测替换。"
            )
        if placeholders and PLACEHOLDER_POLICY == "error":
            raise ValueError(
                f"{identifier}: 发现占位符{sorted(placeholders)}，但PLACEHOLDER_POLICY='error'"
            )
        ids.append(identifier)
        sequences.append(sequence)
        proteins.append(protein_id(identifier))
    labels = read_labels(LABELS_TSV, ids) if LABELS_TSV is not None else None
    return ids, sequences, labels, proteins


def encode_sequences(sequences: list[str], lookup: np.ndarray):
    """对标准残基查表，并显式记录 X 与 - 两种占位符。

    placeholder_type：0=标准氨基酸，1=X，2=-。X 与 - 在 AAindex 数值上同用第21行
    的均值向量；模型若需区分二者，可读取 placeholder_type 或相应掩码，而不是从
    X 本身推断。这不会改变 X 的逐残基 AAindex 维数。
    """
    if PLACEHOLDER_POLICY not in {"mean_with_mask", "error"}:
        raise ValueError("PLACEHOLDER_POLICY只能是 'mean_with_mask' 或 'error'")
    aa_to_index = {aa: i for i, aa in enumerate(AA_ORDER)}
    type_to_code = {"X": 1, "-": 2}
    token_rows, type_rows, normalized = [], [], []
    for sequence in sequences:
        tokens, types, output_sequence = [], [], []
        for amino_acid in sequence:
            if amino_acid in aa_to_index:
                tokens.append(aa_to_index[amino_acid])
                types.append(0)
                output_sequence.append(amino_acid)
            elif amino_acid in type_to_code:
                if PLACEHOLDER_POLICY == "error":
                    raise ValueError(f"发现占位符{amino_acid!r}，但PLACEHOLDER_POLICY='error'")
                tokens.append(20)  # lookup 的最后一行：20种标准残基的均值
                types.append(type_to_code[amino_acid])
                output_sequence.append("X")  # 保持既有 sequences 字段的归一化兼容性
            else:
                raise ValueError(f"不支持的残基{amino_acid!r}；请先通过load_input验证")
        token_rows.append(tokens)
        type_rows.append(types)
        normalized.append("".join(output_sequence))
    token_ids = np.asarray(token_rows, dtype=np.uint8)
    placeholder_type = np.asarray(type_rows, dtype=np.uint8)
    known_mask = placeholder_type == 0
    return lookup[token_ids], known_mask, normalized, placeholder_type


def atomic_npz(path: Path, **arrays):
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(f"拒绝覆盖：{path}")
    # 仅替换本次目标文件，不删除任何旧目录。失败时清理本函数创建的临时文件。
    fd, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    ids, sequences, labels, proteins = load_input()
    targets = [OUTPUT_FILE, METADATA_FILE]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not OVERWRITE_OUTPUTS:
        raise FileExistsError(
            "已有输出，建议修改OUTPUT_FILE；如确认覆盖可设置"
            "OVERWRITE_OUTPUTS=True：\n" + "\n".join(existing)
        )
    input_paths = {INPUT_FASTA.resolve()}
    if LABELS_TSV is not None:
        input_paths.add(LABELS_TSV.resolve())
    if any(path.resolve() in input_paths for path in targets + [AAINDEX_FILE]):
        raise ValueError("输出或数据库路径不能与输入FASTA/标签文件相同")

    records, database_sha256 = read_database()
    table = build_lookup(records, INDEX_IDS, STANDARDIZE)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"使用{len(table['ids'])}个指标；跳过{len(table['skipped'])}个不完整/常量指标。")
    x, known, normalized, placeholder_type = encode_sequences(sequences, table["lookup"])
    replacements = Counter(
        amino_acid for sequence in sequences for amino_acid in sequence
        if amino_acid in PLACEHOLDER_SYMBOLS
    )
    if not np.isfinite(x).all():
        raise RuntimeError("特征出现NaN/Inf，停止保存")

    arrays = {
        "X": x,
        "ids": np.asarray(ids, dtype=str),
        "protein_ids": np.asarray(proteins, dtype=str),
        "sequences": np.asarray(normalized, dtype=str),
        "raw_sequences": np.asarray(sequences, dtype=str),
        "known_residue_mask": known,
        "placeholder_mask": ~known,
        "x_placeholder_mask": placeholder_type == 1,
        "gap_placeholder_mask": placeholder_type == 2,
        "placeholder_type": placeholder_type,  # 0=canonical, 1=X, 2=-
        "placeholder_type_labels": np.asarray(["canonical", "X", "-"], dtype=str),
        "known_residue_count": known.sum(axis=1, dtype=np.int16),
        "known_residue_fraction": known.mean(axis=1, dtype=np.float32),
        "index_ids": np.asarray(table["ids"], dtype=str),
        "index_descriptions": np.asarray(table["descriptions"], dtype=str),
        "aa_order": np.asarray(list(AA_ORDER + "X"), dtype=str),
        "lookup_table": table["lookup"],
        "raw_standard_aa_table": table["raw"],
        "index_mean": table["mean"],
        "index_std": table["std"],
        "standardized": np.asarray(STANDARDIZE),
        "center_position_1_based": np.asarray(CENTER_POSITION),
        "database_sha256": np.asarray(database_sha256),
    }
    if labels is not None:
        arrays["y"] = labels
        arrays["is_labeled_positive"] = labels
        arrays["label_semantics"] = np.asarray(LABEL_SEMANTICS)
    atomic_npz(OUTPUT_FILE, **arrays)

    metadata = {
        "encoding": "AAindex1", "database_url": AAINDEX_URL,
        "database_file": str(AAINDEX_FILE.resolve()), "database_sha256": database_sha256,
        "index_ids": table["ids"], "index_descriptions": table["descriptions"],
        "skipped_indices": table["skipped"], "standardize": STANDARDIZE,
        "normalization": "per-index z-score across 20 canonical amino acids; ddof=0" if STANDARDIZE else "none",
        "unknown_policy": {
            "allowed_placeholders": sorted(PLACEHOLDER_SYMBOLS),
            "policy": PLACEHOLDER_POLICY,
            "encoding": "20-amino-acid AAindex mean; zero after standardization",
            "known_residue_mask": "False for X and -",
            "placeholder_type": "0=canonical, 1=X, 2=-",
            "sequences_field": "X and - both normalized to X for backward compatibility; use raw_sequences and placeholder_type to distinguish them",
        },
        "labels_included": labels is not None,
        "label_semantics": LABEL_SEMANTICS if labels is not None else None,
        "dtype": "float32", "axis_order": ["sample", "residue", "index"],
        "fragment_length": EXPECTED_FRAGMENT_LENGTH, "center_position_1_based": CENTER_POSITION,
        "input_fasta": str(INPUT_FASTA.resolve()),
        "input_fasta_sha256": hashlib.sha256(INPUT_FASTA.read_bytes()).hexdigest(),
        "labels_tsv": str(LABELS_TSV.resolve()) if LABELS_TSV is not None else None,
        "labels_tsv_sha256": hashlib.sha256(LABELS_TSV.read_bytes()).hexdigest()
        if LABELS_TSV is not None else None,
        "output_file": str(OUTPUT_FILE.resolve()),
        "shape": list(x.shape),
        "sample_count": len(ids),
        "positive_count": int(labels.sum()) if labels is not None else None,
        "unlabeled_count": int(len(labels) - labels.sum()) if labels is not None else None,
        "placeholder_counts": dict(replacements),
        "samples_with_any_placeholder": int((~known).any(axis=1).sum()),
        "samples_with_center_placeholder": int((~known[:, CENTER_POSITION - 1]).sum()),
        "input_order_preserved": True,
        "protein_id_format_checked": REQUIRE_PROTEIN_ID_FORMAT,
        "complete": True,
    }
    with METADATA_FILE.open("w" if OVERWRITE_OUTPUTS else "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"完成：X={x.shape}，占位符X/-总数={sum(replacements.values())}，"
          f"含占位符样本={int((~known).any(axis=1).sum())}")
    if labels is not None:
        print(f"标签：P={int(labels.sum())}，U={int(len(labels) - labels.sum())}")
    else:
        print("未提供LABELS_TSV：NPZ中未保存y，程序没有猜测标签。")
    print(f"特征文件：{OUTPUT_FILE.resolve()}")
    print(f"元数据：{METADATA_FILE.resolve()}")
    print("融合其他编码前，务必按ids核对并对齐样本。")


if __name__ == "__main__":
    main()
