import argparse
import json
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset


DEFAULT_DATASETS = [
    # Good general English prose
    "wikitext:wikitext-103-v1:train:text",
    # Long-form books (English) with clear sentence-like text
    "bookcorpus:plain_text:train:text",
]


def parse_dataset_spec(spec: str):
    """
    spec format: dataset_name[:config][:split][:text_field]
    examples:
      wikitext:wikitext-103-v1:train:text
      ag_news::train:description
      oscar:unshuffled_deduplicated_en:train:text
    """
    parts = spec.split(":")
    while len(parts) < 4:
        parts.append("")
    name, config, split, text_field = parts[:4]
    return name, (config or None), (split or "train"), (text_field or None)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_english(text: str) -> bool:
    if not text:
        return False
    letters = re.findall(r"[A-Za-z]", text)
    if len(letters) < 8:
        return False
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / max(1, len(text))
    return ascii_ratio >= 0.90


def sentence_candidates(blob: str) -> Iterable[str]:
    # Split on sentence boundaries; keep it lightweight and dependency-free
    for part in re.split(r"(?<=[.!?])\s+", blob):
        s = normalize_text(part)
        if s:
            yield s


# Patterns that indicate noisy / non-prose text
_NOISE_PATTERNS = [
    re.compile(r"^=+"),                          # wiki section headers
    re.compile(r"==[^=].*=="),                   # inline wiki headers
    re.compile(r"\b\d+\s+at\s+(dot|dash)\s+at\b", re.I),  # number artifacts
    re.compile(r"^(?:[A-Z][^.!?]*,\s*){1,2}\d{4}[.,]?$"),  # bibliography lines
    re.compile(r"\bunk\b", re.I),                # unexpanded unknowns
    re.compile(r"(?:^|\s)={1,}\s"),             # stray equals
    re.compile(r"<[^>]+>"),                       # HTML tags
    re.compile(r"\[\[.*\]\]"),                   # wiki links
    re.compile(r"http[s]?://"),                  # URLs
    re.compile(r"\|\s*\|"),                      # table syntax
    re.compile(r"^\W+$"),                         # no real words
]


def is_noisy(text: str) -> bool:
    for pat in _NOISE_PATTERNS:
        if pat.search(text):
            return True
    words = text.split()
    if len(words) < 7:
        return True
    # Too many numeric tokens (> 30% of words)
    num_count = sum(1 for w in words if re.fullmatch(r"[\d.,/:%-]+", w))
    if num_count / max(1, len(words)) > 0.30:
        return True
    # Excessive capitalised words in a row (likely a proper-noun list)
    cap_runs = re.findall(r"(?:[A-Z][a-z]+\s+){4,}", text)
    if cap_runs:
        return True
    return False


def valid_sentence(text: str, min_chars: int, max_chars: int) -> bool:
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if text.count(" ") < 6:
        return False
    return True


def extract_text_field(example: Dict, preferred_field: Optional[str]) -> Optional[str]:
    if preferred_field and preferred_field in example and isinstance(example[preferred_field], str):
        return example[preferred_field]

    common = ["text", "sentence", "content", "article", "description", "summary", "review"]
    for key in common:
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val

    for _, val in example.items():
        if isinstance(val, str) and val.strip():
            return val

    return None


def collect_sentences(
    dataset_specs: Sequence[str],
    target_count: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    max_examples_per_dataset: int,
) -> List[Dict]:
    random.seed(seed)
    items: List[Dict] = []
    seen = set()

    for spec in dataset_specs:
        if len(items) >= target_count:
            break

        name, config, split, field = parse_dataset_spec(spec)
        print(f"[load] {name} | config={config} | split={split} | field={field}")

        try:
            ds = load_dataset(name, config, split=split, streaming=True)
        except Exception as exc:
            print(f"  [skip] could not load {spec}: {exc}")
            continue

        consumed = 0
        kept = 0
        for ex in ds:
            consumed += 1
            if consumed > max_examples_per_dataset or len(items) >= target_count:
                break

            raw = extract_text_field(ex, field)
            if not raw:
                continue

            for sent in sentence_candidates(raw):
                if not looks_english(sent):
                    continue
                if not valid_sentence(sent, min_chars, max_chars):
                    continue
                if is_noisy(sent):
                    continue
                key = sent.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append({"text": sent, "source": name})
                kept += 1
                if len(items) >= target_count:
                    break

        print(f"  [ok] consumed={consumed}, added={kept}, total={len(items)}")

    random.shuffle(items)
    return items[:target_count]


def save_outputs(rows: Sequence[Dict], out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{base_name}.txt")
    jsonl_path = os.path.join(out_dir, f"{base_name}.jsonl")

    with open(txt_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["text"] + "\n")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[saved] {txt_path}")
    print(f"[saved] {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Download English text from Hugging Face datasets.")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/tts_text_en", help="Output folder")
    parser.add_argument("--base_name", type=str, default="english_sentences", help="Output filename prefix")
    parser.add_argument("--target_count", type=int, default=1000, help="Number of sentences to collect")
    parser.add_argument("--min_chars", type=int, default=50, help="Minimum sentence length")
    parser.add_argument("--max_chars", type=int, default=220, help="Maximum sentence length")
    parser.add_argument("--max_examples_per_dataset", type=int, default=20000, help="Safety cap per dataset")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=(
            "Dataset specs in format name[:config][:split][:text_field]. "
            "Example: wikitext:wikitext-103-v1:train:text"
        ),
    )

    args = parser.parse_args()

    rows = collect_sentences(
        dataset_specs=args.datasets,
        target_count=args.target_count,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_examples_per_dataset=args.max_examples_per_dataset,
    )

    if not rows:
        raise SystemExit("No sentences collected. Try a different dataset spec.")

    save_outputs(rows, args.out_dir, args.base_name)
    print(f"[done] collected={len(rows)}")


if __name__ == "__main__":
    main()
