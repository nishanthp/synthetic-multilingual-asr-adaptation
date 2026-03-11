from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


TEXT_FIELD = "text"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SPEAKER = "Ryan"


@dataclass(frozen=True)
class PipelineConfig:
    dataset_name: str = "karpathy/tinystories-gpt4-clean"
    dataset_config: Optional[str] = None
    split: str = "train"
    output_root: str = "generated_speech_dataset"
    instruct: str = ""
    max_samples: int = 3
    offset: int = 0
    max_characters: int = 5000
    use_gpu: bool = True
    streaming: bool = True
    overwrite: bool = False


def download_text_data(
    dataset_name: str = "karpathy/tinystories-gpt4-clean",
    dataset_config: Optional[str] = None,
    split: str = "train",
    streaming: bool = True,
    offset: int = 0,
    limit: Optional[int] = None,
):
    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)

    if streaming:
        if offset:
            dataset = dataset.skip(offset)
        if limit is not None:
            dataset = dataset.take(limit)
        return dataset

    if limit is None:
        return dataset.select(range(offset, len(dataset)))

    end_index = min(offset + limit, len(dataset))
    return dataset.select(range(offset, end_index))


def synthesize_with_qwen_tts(
    text: str,
    output_path: str,
    model: Any,
    language: str,
    speaker: str,
    instruct: str,
) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )
    sf.write(destination, wavs[0], sample_rate)
    return str(destination)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "dataset"


def normalize_text(text: str, max_characters: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_characters:
        return compact

    truncated = compact[:max_characters].rsplit(" ", 1)[0].strip()
    return truncated or compact[:max_characters].strip()


def extract_text_records(
    dataset: Iterable[dict[str, Any]],
    max_samples: int,
    max_characters: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for source_index, example in enumerate(dataset):
        raw_text = example.get(TEXT_FIELD)
        if not isinstance(raw_text, str):
            continue

        text = normalize_text(raw_text, max_characters=max_characters)
        if not text:
            continue

        records.append(
            {
                "sample_id": f"sample_{len(records):05d}",
                "source_index": source_index,
                "text": text,
            }
        )

        if len(records) >= max_samples:
            break

    if not records:
        raise ValueError("No usable text rows were extracted from the dataset.")

    return records


def prepare_output_dir(config: PipelineConfig) -> Path:
    dataset_slug = slugify(config.dataset_name)
    split_slug = slugify(config.split)
    root = Path(config.output_root) / dataset_slug / split_slug
    wav_dir = root / "wavs"

    if config.overwrite and root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    wav_dir.mkdir(parents=True, exist_ok=True)
    return root


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_speech_dataset(config: PipelineConfig) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required. Install it with `pip install torch`.") from exc

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError(
            "The `qwen-tts` package is required. Install it with `pip install qwen-tts`."
        ) from exc

    if config.use_gpu and not torch.cuda.is_available():
        raise RuntimeError("`--use-gpu` was provided, but CUDA is not available on this machine.")

    dataset = download_text_data(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.split,
        streaming=config.streaming,
        offset=config.offset,
        limit=config.max_samples,
    )
    records = extract_text_records(
        dataset=dataset,
        max_samples=config.max_samples,
        max_characters=config.max_characters,
    )

    output_dir = prepare_output_dir(config)
    wav_dir = output_dir / "wavs"
    manifest_path = output_dir / "manifest.jsonl"
    config_path = output_dir / "run_config.json"
    text_path = output_dir / "texts.jsonl"

    language = "English"
    dtype = torch.bfloat16 if config.use_gpu else torch.float32
    load_kwargs: dict[str, Any] = {
        "device_map": "cuda:0" if config.use_gpu else "cpu",
        "dtype": dtype,
        "attn_implementation": "sdpa",
    }

    tts_model = Qwen3TTSModel.from_pretrained(DEFAULT_MODEL_NAME, **load_kwargs)

    manifest_rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc="Synthesizing", unit="sample"):
        wav_name = f"{record['sample_id']}.wav"
        wav_path = wav_dir / wav_name

        synthesize_with_qwen_tts(
            text=record["text"],
            output_path=str(wav_path),
            model=tts_model,
            language=language,
            speaker=DEFAULT_SPEAKER,
            instruct=config.instruct,
        )

        manifest_rows.append(
            {
                "sample_id": record["sample_id"],
                "source_index": record["source_index"],
                "text": record["text"],
                "audio_filepath": str(wav_path.relative_to(output_dir)),
                "language": language,
                "dataset_name": config.dataset_name,
                "dataset_config": config.dataset_config,
                "split": config.split,
                "text_field": TEXT_FIELD,
                "tts_model": DEFAULT_MODEL_NAME,
                "speaker": DEFAULT_SPEAKER,
                "instruct": config.instruct,
            }
        )

    write_jsonl(text_path, records)
    write_jsonl(manifest_path, manifest_rows)
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    config = PipelineConfig()
    output_dir = build_speech_dataset(config)
    print(f"Saved dataset to: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
