#!/usr/bin/env python3
"""使用ProtT5对单个FASTA文件编码，并增量保存为HDF5文件。

在IDE中修改“IDE运行配置区”后直接点击运行，不需要命令行参数。

每个文件保存：
1. residue_embeddings: (N, 21, 1024)，逐残基嵌入；
2. center_embeddings:  (N, 1024)，中心候选位点嵌入；
3. mean_embeddings:    (N, 1024)，片段平均池化嵌入；
4. ids和sequences，以及可选的labels，供后续识别模型使用。

程序按批次直接写入磁盘，不会将全部ProtT5特征同时保存在内存中；
意外中断后可从HDF5文件记录的批次位置继续运行。

窗口两端的 X 或 - 可作为占位符继续编码：- 会转换为 ProtT5 可识别的 X token，
但会保存原始序列、占位符类型和已知残基掩码；mean_embeddings 仅池化真实氨基酸。
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


# ========================== IDE运行配置区 ==========================
# 每次只编码一个FASTA。依次处理02～05时，同步修改下面三个文件名。
INPUT_FASTA = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test.fasta"
)

# 混合P/U文件可配置配套标签表，表中必须有sample_id和label两列。
# 若只需要编码、或输入FASTA没有可靠标签，请设置为None；程序不会根据文件名猜标签。
LABELS_TSV: Path | None = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_label.tsv"
)

OUTPUT_FILE = Path(
    r"E:\olg\O-linkedGlycosylation\data\val_test\val_test_prott5.h5"
)
METADATA_FILE = OUTPUT_FILE.with_name(OUTPUT_FILE.stem + "_metadata.json")

# 官方ProtT5半精度、仅编码器模型。也可以改成本地模型文件夹路径。
MODEL_NAME_OR_PATH = "Rostlab/prot_t5_xl_half_uniref50-enc"

EXPECTED_FRAGMENT_LENGTH = 21
CENTER_POSITION = 11  # 从1开始计数；Python数组索引为CENTER_POSITION - 1

# auto：有NVIDIA CUDA时使用GPU，否则使用CPU。
DEVICE = "auto"  # 可选："auto"、"cuda"、"cpu"

# ProtT5-XL在CPU上运行非常慢且会占用大量内存。False表示CUDA异常时立即
# 报错，避免程序悄悄退回CPU；确实需要CPU运行时才改为True。
ALLOW_CPU_FALLBACK = False

# 21肽较短，可根据显存增加批大小；CUDA显存不足时程序会自动减半重试。
GPU_BATCH_SIZE = 16
CPU_BATCH_SIZE = 1

# float16可显著减少保存空间；如后续确需更高精度，可改成"float32"。
STORAGE_DTYPE = "float16"  # 可选："float16"、"float32"

# 是否保存体积较大的逐残基嵌入。建议保留，方便后续CNN/注意力模型使用。
SAVE_RESIDUE_EMBEDDINGS = True

# 保持FASTA顺序不变。融合不同编码时仍必须按ids核对，不能只按行号假定一致。

# 若输出文件已存在且未完成，则从中断位置继续；已完成则跳过。
RESUME_IF_POSSIBLE = True

# ProtT5 分词器不能接收 -。X 和 - 均作为占位符映射为 token X，保留原窗口长度；
# 但会通过 known_residue_mask 和 placeholder_type 区分。B/Z/U/O/J 等仍报错，
# 避免将生物学含义不确定的字符静默替换。
PLACEHOLDER_SYMBOLS = frozenset({"X", "-"})
PLACEHOLDER_POLICY = "map_to_X_with_mask"
# ==================================================================


STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    """读取多行FASTA，返回第一段标题作为ID以及大写序列。"""
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
                        f"{path} 第{line_number}行在第一个FASTA标题前出现序列"
                    )
                sequence_parts.append("".join(line.split()))

    if header is not None:
        sample_ids.append(header)
        sequences.append("".join(sequence_parts).upper())

    if not sample_ids:
        raise ValueError(f"{path} 中没有找到FASTA记录")
    if any(not sequence for sequence in sequences):
        raise ValueError(f"{path} 中至少存在一条空序列")

    duplicates = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"{path} 中存在重复样本ID：{duplicates[:5]}")
    return sample_ids, sequences


def normalize_for_prott5(sequence: str) -> tuple[str, np.ndarray, Counter[str]]:
    """将 X/- 映射为 ProtT5 token X，并返回占位符类型。

    placeholder_type：0=标准氨基酸，1=原始字符 X，2=原始字符 -。
    对 ProtT5 而言二者均输入 X；原始差异由 HDF5 的 placeholder_type 保留。
    """
    if PLACEHOLDER_POLICY != "map_to_X_with_mask":
        raise ValueError("PLACEHOLDER_POLICY 当前仅支持 'map_to_X_with_mask'")
    normalized: list[str] = []
    placeholder_type = np.zeros(len(sequence), dtype=np.uint8)
    replacements: Counter[str] = Counter()
    for position, residue in enumerate(sequence.upper()):
        if residue in STANDARD_AMINO_ACIDS:
            normalized.append(residue)
        elif residue == "X":
            normalized.append("X")
            placeholder_type[position] = 1
            replacements[residue] += 1
        elif residue == "-":
            normalized.append("X")  # - 不在 ProtT5 残基词表中
            placeholder_type[position] = 2
            replacements[residue] += 1
        else:
            raise ValueError(
                f"发现不支持的残基 {residue!r}；仅允许20种标准氨基酸、X或-。"
                "B/Z/U/O/J不会被自动猜测替换。"
            )
    return "".join(normalized), placeholder_type, replacements


def read_labels(path: Path, fasta_ids: list[str]) -> np.ndarray:
    """读取TSV并按sample_id对齐；标签表顺序可以与FASTA不同。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到标签文件：{path}")
    import csv
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


def load_one_file() -> tuple[
    list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray | None, Counter[str]
]:
    """读取一个 FASTA，并生成 ProtT5 token 序列及 X/- 掩码。"""
    sample_ids, raw_sequences = read_fasta(INPUT_FASTA)
    normalized_sequences: list[str] = []
    type_rows: list[np.ndarray] = []
    replacement_counts: Counter[str] = Counter()
    for sample_id, sequence in zip(sample_ids, raw_sequences):
        if len(sequence) != EXPECTED_FRAGMENT_LENGTH:
            raise ValueError(
                f"样本{sample_id}长度为{len(sequence)}，"
                f"预期为{EXPECTED_FRAGMENT_LENGTH}"
            )
        normalized, placeholder_type, replacements = normalize_for_prott5(sequence)
        normalized_sequences.append(normalized)
        type_rows.append(placeholder_type)
        replacement_counts.update(replacements)

    placeholder_types = np.stack(type_rows).astype(np.uint8, copy=False)
    known_mask = placeholder_types == 0
    labels = read_labels(LABELS_TSV, sample_ids) if LABELS_TSV is not None else None
    return (sample_ids, raw_sequences, normalized_sequences, known_mask,
            placeholder_types, labels, replacement_counts)


def select_device() -> torch.device:
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if not ALLOW_CPU_FALLBACK:
            raise RuntimeError(
                "PyTorch未能启用CUDA，程序已停止以避免ProtT5-XL退回CPU。\n"
                f"当前PyTorch版本：{torch.__version__}\n"
                f"PyTorch编译CUDA版本：{torch.version.cuda}\n"
                "请更新NVIDIA驱动，或安装与当前驱动兼容的PyTorch；修复后先确认\n"
                "python -c \"import torch; print(torch.cuda.is_available())\"\n"
                "输出True。若确实要使用CPU，请设置ALLOW_CPU_FALLBACK = True。"
            )
        return torch.device("cpu")
    if DEVICE not in {"cuda", "cpu"}:
        raise ValueError('DEVICE只能是"auto"、"cuda"或"cpu"')
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用CUDA，但PyTorch未检测到可用的NVIDIA GPU")
    return torch.device(DEVICE)


def check_runtime_dependencies() -> None:
    """在下载大模型之前给出缺失依赖的明确安装命令。"""
    missing: list[str] = []
    for module_name, package_name in (
        ("sentencepiece", "sentencepiece"),
        ("google.protobuf", "protobuf"),
    ):
        try:
            available = importlib.util.find_spec(module_name) is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            missing.append(package_name)

    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "ProtT5运行依赖缺失：" + ", ".join(missing) + "\n"
            "请在当前IDE所用解释器中执行：\n"
            f'"{sys.executable}" -m pip install -U {packages}\n'
            "安装后请完全重启IDE再运行。"
        )


def load_tokenizer_and_model(
    device: torch.device,
) -> tuple[T5Tokenizer, T5EncoderModel]:
    print(f"正在加载ProtT5：{MODEL_NAME_OR_PATH}")
    print("首次运行需要从Hugging Face下载数GB模型文件，请耐心等待。")

    # legacy=True保持ProtTrans官方示例所使用的T5分词行为。
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        do_lower_case=False,
        legacy=True,
    )

    # GPU上使用半精度；CPU上转换为float32以获得更广泛的算子兼容性。
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = T5EncoderModel.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=model_dtype,
    )
    model.to(device)
    model.eval()

    hidden_size = int(model.config.d_model)
    print(
        f"模型加载完成：device={device}，model_dtype={model_dtype}，"
        f"hidden_size={hidden_size}"
    )
    if device.type == "cpu":
        print("警告：ProtT5-XL在CPU上非常慢且需要较大内存，推荐使用NVIDIA GPU。")
    return tokenizer, model


def strings_equal(dataset: h5py.Dataset, values: list[str]) -> bool:
    stored = dataset.asstr()[:]
    return np.array_equal(stored, np.asarray(values, dtype=str))


def create_or_open_output(
    path: Path,
    sample_ids: list[str],
    sequences: list[str],
    raw_sequences: list[str],
    known_mask: np.ndarray,
    placeholder_type: np.ndarray,
    labels: np.ndarray | None,
    hidden_size: int,
    chunk_size: int,
) -> tuple[h5py.File, int]:
    """创建HDF5结果；若是兼容的未完成文件，则从断点继续。"""
    storage_dtype = np.dtype(STORAGE_DTYPE)
    sample_count = len(sample_ids)
    expected_shape = (sample_count, EXPECTED_FRAGMENT_LENGTH, hidden_size)

    if path.exists():
        if not RESUME_IF_POSSIBLE:
            raise FileExistsError(
                f"输出文件已存在：{path}。请改OUTPUT_FILE或启用断点续跑。"
            )
        handle = h5py.File(path, "r+")
        try:
            required_attrs = {
                "model_name_or_path": str(MODEL_NAME_OR_PATH),
                "fragment_length": EXPECTED_FRAGMENT_LENGTH,
                "hidden_size": hidden_size,
                "storage_dtype": STORAGE_DTYPE,
                "save_residue_embeddings": bool(SAVE_RESIDUE_EMBEDDINGS),
                "sample_count": sample_count,
                "labels_included": labels is not None,
                "placeholder_policy": PLACEHOLDER_POLICY,
            }
            for key, expected in required_attrs.items():
                if key not in handle.attrs or handle.attrs[key] != expected:
                    raise ValueError(
                        f"已有文件属性{key}不兼容："
                        f"已有={handle.attrs.get(key)!r}，当前={expected!r}"
                    )
            if not strings_equal(handle["ids"], sample_ids):
                raise ValueError("已有文件的样本ID或顺序与当前输入不一致")
            required_datasets = {"original_sequences", "known_residue_mask", "placeholder_type"}
            missing = required_datasets - set(handle.keys())
            if missing:
                raise ValueError(
                    f"已有HDF5缺少新版占位符字段{sorted(missing)}；请使用新的OUTPUT_FILE，"
                    "不要把新旧编码结果混在同一文件。"
                )
            if not strings_equal(handle["sequences"], sequences):
                raise ValueError("已有文件的ProtT5 token序列或顺序与当前输入不一致")
            if not strings_equal(handle["original_sequences"], raw_sequences):
                raise ValueError("已有文件的原始序列或顺序与当前输入不一致")
            if not np.array_equal(handle["known_residue_mask"][:].astype(bool), known_mask):
                raise ValueError("已有文件的已知残基掩码与当前输入不一致")
            if not np.array_equal(handle["placeholder_type"][:], placeholder_type):
                raise ValueError("已有文件的占位符类型与当前输入不一致")
            if labels is not None and not np.array_equal(handle["labels"][:], labels):
                raise ValueError("已有文件的标签与当前输入不一致")
            if labels is None and "labels" in handle:
                raise ValueError("已有文件包含标签，但当前配置LABELS_TSV=None")
            if handle["center_embeddings"].shape != (sample_count, hidden_size):
                raise ValueError("已有文件的中心嵌入形状不兼容")
            if handle["mean_embeddings"].shape != (sample_count, hidden_size):
                raise ValueError("已有文件的平均嵌入形状不兼容")
            if SAVE_RESIDUE_EMBEDDINGS and handle["residue_embeddings"].shape != expected_shape:
                raise ValueError("已有文件的逐残基嵌入形状不兼容")

            completed = int(handle.attrs.get("completed_samples", 0))
            if not 0 <= completed <= sample_count:
                raise ValueError("已有文件的断点位置无效")
            return handle, completed
        except Exception:
            handle.close()
            raise

    handle = h5py.File(path, "w")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    vector_chunks = (min(chunk_size, sample_count), hidden_size)

    if labels is not None:
        handle.create_dataset("labels", data=labels, dtype=np.uint8)
    handle.create_dataset(
        "ids", data=np.asarray(sample_ids, dtype=object), dtype=string_dtype
    )
    handle.create_dataset(
        "sequences", data=np.asarray(sequences, dtype=object), dtype=string_dtype
    )
    handle.create_dataset(
        "original_sequences", data=np.asarray(raw_sequences, dtype=object), dtype=string_dtype
    )
    handle.create_dataset("known_residue_mask", data=known_mask.astype(np.uint8), dtype=np.uint8)
    handle.create_dataset("placeholder_type", data=placeholder_type, dtype=np.uint8)
    handle.create_dataset("known_residue_count", data=known_mask.sum(axis=1, dtype=np.int16), dtype=np.int16)
    handle.create_dataset(
        "known_residue_fraction", data=known_mask.mean(axis=1, dtype=np.float32), dtype=np.float32
    )
    handle.create_dataset(
        "center_embeddings",
        shape=(sample_count, hidden_size),
        dtype=storage_dtype,
        chunks=vector_chunks,
        compression="lzf",
    )
    handle.create_dataset(
        "mean_embeddings",
        shape=(sample_count, hidden_size),
        dtype=storage_dtype,
        chunks=vector_chunks,
        compression="lzf",
    )
    if SAVE_RESIDUE_EMBEDDINGS:
        handle.create_dataset(
            "residue_embeddings",
            shape=expected_shape,
            dtype=storage_dtype,
            chunks=(min(chunk_size, sample_count), EXPECTED_FRAGMENT_LENGTH, hidden_size),
            compression="lzf",
        )

    handle.attrs.update(
        {
            "model_name_or_path": str(MODEL_NAME_OR_PATH),
            "fragment_length": EXPECTED_FRAGMENT_LENGTH,
            "center_position_1_based": CENTER_POSITION,
            "hidden_size": hidden_size,
            "storage_dtype": STORAGE_DTYPE,
            "save_residue_embeddings": bool(SAVE_RESIDUE_EMBEDDINGS),
            "sample_count": sample_count,
            "labels_included": labels is not None,
            "label_semantics": (
                "1=labeled_positive(P); 0=unlabeled(U), NOT verified_negative"
                if labels is not None else "not_provided"
            ),
            "placeholder_policy": PLACEHOLDER_POLICY,
            "placeholder_symbols": "X,-",
            "placeholder_type_definition": "0=canonical amino acid, 1=X, 2=-",
            "mean_pooling": "mean over canonical amino-acid positions only; zero vector if none",
            "input_order_preserved": True,
            "completed_samples": 0,
            "complete": False,
        }
    )
    handle.flush()
    return handle, 0


def tokenize_batch(
    tokenizer: T5Tokenizer, sequences: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # ProtT5要求残基之间使用空格分隔。
    spaced_sequences = [" ".join(sequence) for sequence in sequences]
    tokens = tokenizer(
        spaced_sequences,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # 每条固定长度序列应为L个残基token加1个EOS；padding不计入。
    token_lengths = attention_mask.sum(dim=1).cpu().numpy()
    expected_token_length = EXPECTED_FRAGMENT_LENGTH + 1
    if not np.all(token_lengths == expected_token_length):
        raise RuntimeError(
            "ProtT5分词结果与残基长度不是一一对应："
            f"期望有效token数={expected_token_length}，实际={token_lengths.tolist()}"
        )
    return input_ids.to(device), attention_mask.to(device)


def encode_file(
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> dict[str, int | str]:
    (sample_ids, raw_sequences, sequences, known_mask,
     placeholder_type, labels, replacements) = load_one_file()
    sample_count = len(sample_ids)
    hidden_size = int(model.config.d_model)
    initial_batch_size = GPU_BATCH_SIZE if device.type == "cuda" else CPU_BATCH_SIZE
    if initial_batch_size < 1:
        raise ValueError("批大小必须至少为1")

    handle, start_index = create_or_open_output(
        OUTPUT_FILE,
        sample_ids,
        sequences,
        raw_sequences,
        known_mask,
        placeholder_type,
        labels,
        hidden_size,
        initial_batch_size,
    )

    if bool(handle.attrs.get("complete", False)):
        print(f"文件已完成，跳过：{OUTPUT_FILE}")
        handle.close()
        return {
            "samples": sample_count,
            "positive": int(labels.sum()) if labels is not None else -1,
            "unlabeled": int(sample_count - labels.sum()) if labels is not None else -1,
            "output_file": OUTPUT_FILE.name,
            "replacements": json.dumps(dict(replacements), ensure_ascii=False),
            "placeholder_samples": int((~known_mask).any(axis=1).sum()),
        }

    print(
        f"\n开始编码：样本数={sample_count}，"
        f"从索引{start_index}继续，批大小={initial_batch_size}"
    )
    batch_size = initial_batch_size
    current = start_index
    started_at = time.time()

    try:
        while current < sample_count:
            end = min(current + batch_size, sample_count)
            batch_sequences = sequences[current:end]

            try:
                input_ids, attention_mask = tokenize_batch(
                    tokenizer, batch_sequences, device
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    # 前L个token对应L个残基；排除末尾EOS和padding。
                    residue_tensor = output.last_hidden_state[
                        :, :EXPECTED_FRAGMENT_LENGTH, :
                    ]
                    center_tensor = residue_tensor[:, CENTER_POSITION - 1, :]
                    # 两端 X/- 不参与均值池化；若一条序列全是占位符，则返回零向量而非 NaN。
                    batch_known = torch.from_numpy(known_mask[current:end]).to(
                        device=device, dtype=torch.float32
                    )
                    denominator = batch_known.sum(dim=1, keepdim=True)
                    mean_tensor = (residue_tensor.float() * batch_known.unsqueeze(-1)).sum(dim=1)
                    mean_tensor = torch.where(
                        denominator > 0,
                        mean_tensor / denominator.clamp_min(1.0),
                        torch.zeros_like(mean_tensor),
                    )

                residue_array = residue_tensor.detach().cpu().numpy().astype(
                    STORAGE_DTYPE, copy=False
                )
                center_array = center_tensor.detach().cpu().numpy().astype(
                    STORAGE_DTYPE, copy=False
                )
                mean_array = mean_tensor.detach().cpu().numpy().astype(
                    STORAGE_DTYPE, copy=False
                )

                if SAVE_RESIDUE_EMBEDDINGS:
                    handle["residue_embeddings"][current:end] = residue_array
                handle["center_embeddings"][current:end] = center_array
                handle["mean_embeddings"][current:end] = mean_array
                handle.attrs["completed_samples"] = end
                handle.flush()

                current = end
                elapsed = max(time.time() - started_at, 1e-9)
                processed = current - start_index
                speed = processed / elapsed
                remaining = (sample_count - current) / speed if speed > 0 else 0
                print(
                    f"\r{current}/{sample_count} "
                    f"({100 * current / sample_count:5.1f}%)，"
                    f"约剩余{remaining / 60:.1f}分钟",
                    end="",
                    flush=True,
                )

                del input_ids, attention_mask, output
                del residue_tensor, center_tensor, mean_tensor, batch_known, denominator
                del residue_array, center_array, mean_array

            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda" or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                gc.collect()
                print(f"\nCUDA显存不足，批大小自动降低为{batch_size}并重试。")

        handle.attrs["complete"] = True
        handle.flush()
        print(f"\n编码完成：{OUTPUT_FILE}")
    finally:
        handle.close()

    return {
        "samples": sample_count,
        "positive": int(labels.sum()) if labels is not None else -1,
        "unlabeled": int(sample_count - labels.sum()) if labels is not None else -1,
        "output_file": OUTPUT_FILE.name,
        "replacements": json.dumps(dict(replacements), ensure_ascii=False),
        "placeholder_samples": int((~known_mask).any(axis=1).sum()),
    }


def write_metadata(hidden_size: int, device: torch.device, summary: dict) -> None:
    metadata = {
        "encoding": "ProtT5",
        "model_name_or_path": str(MODEL_NAME_OR_PATH),
        "hidden_size": hidden_size,
        "fragment_length": EXPECTED_FRAGMENT_LENGTH,
        "center_position_1_based": CENTER_POSITION,
        "storage_dtype": STORAGE_DTYPE,
        "save_residue_embeddings": SAVE_RESIDUE_EMBEDDINGS,
        "device_used": str(device),
        "placeholder_policy": {
            "symbols": ["X", "-"],
            "tokenization": "both X and - are sent to ProtT5 as token X",
            "stored_masks": ["known_residue_mask", "placeholder_type"],
            "placeholder_type": "0=canonical, 1=X, 2=-",
            "mean_pooling": "canonical residues only; zero vector if no canonical residues",
        },
        "input_fasta": str(INPUT_FASTA.resolve()),
        "input_fasta_sha256": hashlib.sha256(INPUT_FASTA.read_bytes()).hexdigest(),
        "labels_tsv": str(LABELS_TSV.resolve()) if LABELS_TSV is not None else None,
        "labels_tsv_sha256": (
            hashlib.sha256(LABELS_TSV.read_bytes()).hexdigest()
            if LABELS_TSV is not None else None
        ),
        "labels_included": LABELS_TSV is not None,
        "label_semantics": (
            "1=labeled_positive(P); 0=unlabeled(U), NOT verified_negative"
            if LABELS_TSV is not None else None
        ),
        "input_order_preserved": True,
        "output_file": str(OUTPUT_FILE.resolve()),
        "summary": summary,
    }
    with METADATA_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    if not 1 <= CENTER_POSITION <= EXPECTED_FRAGMENT_LENGTH:
        raise ValueError("CENTER_POSITION必须位于片段范围内")
    if STORAGE_DTYPE not in {"float16", "float32"}:
        raise ValueError('STORAGE_DTYPE只能是"float16"或"float32"')

    # 先验证输入和标签，再下载/加载数GB模型。
    load_one_file()
    check_runtime_dependencies()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f"计算设备：{device}")
    tokenizer, model = load_tokenizer_and_model(device)
    summary = encode_file(tokenizer, model, device)
    write_metadata(int(model.config.d_model), device, summary)

    print(f"\n本文件编码完成：{OUTPUT_FILE.resolve()}")
    print("HDF5数据集：ids、sequences、original_sequences、known_residue_mask、placeholder_type、center_embeddings、mean_embeddings")
    if LABELS_TSV is not None:
        print("同时包含：labels（1=P，0=U，不是已验证负例）")
    if SAVE_RESIDUE_EMBEDDINGS:
        print("同时包含：residue_embeddings")


if __name__ == "__main__":
    main()
