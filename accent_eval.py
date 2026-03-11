import math
from collections import defaultdict
from dataclasses import dataclass

from datasets import Audio, load_dataset
from jiwer import wer
from transformers import pipeline

# ============================================================
# Accent Evaluation Script (Whisper + Hugging Face dataset)
# ============================================================
# Purpose:
#   1) Load an accent/country speech dataset from Hugging Face
#   2) Transcribe clips with Whisper
#   3) Compute WER by group (country/accent/region)
#
# Why this is useful:
#   It helps answer questions like:
#   "Does ASR perform worse for certain accents/countries?"
#
# Notes:
#   - This script is intentionally verbose + heavily commented.
#   - It is defensive about schema differences across datasets.
#   - If the default dataset fields do not match, adjust the field lists below.
# ============================================================


# ---------------------------
# Configuration
# ---------------------------
# Try this first (accent-focused dataset):
#   changelinglab/speechaccentarchive-pr
# Fallback option:
#   mozilla-foundation/common_voice_17_0 (config: "en")
DATASET_ID = "changelinglab/speechaccentarchive-pr"
DATASET_CONFIG = None
DATASET_SPLIT = "test"

# Keep this small initially so the run finishes quickly.
MAX_SAMPLES = 20

# Whisper model (small is a good speed/quality balance on CPU)
ASR_MODEL_ID = "openai/whisper-small"

# If True, we only use rows where both text+audio are present.
STRICT_ROWS_ONLY = True


# ---------------------------
# Candidate field names
# ---------------------------
# Different datasets call columns different names.
# We try these in order and pick the first available.
AUDIO_FIELD_CANDIDATES = ["audio", "wav", "speech"]
TEXT_FIELD_CANDIDATES = ["ref_text", "sentence", "text", "transcript", "transcription"]
GROUP_FIELD_CANDIDATES = ["country", "accent", "region", "locale", "native_language"]


@dataclass
class FieldMap:
    audio: str
    text: str
    group: str


def pick_first_existing(columns, candidates):
    """Return the first candidate column that exists in dataset columns."""
    for c in candidates:
        if c in columns:
            return c
    return None


def detect_fields(columns):
    """Auto-detect audio/text/group fields from dataset columns."""
    audio_field = pick_first_existing(columns, AUDIO_FIELD_CANDIDATES)
    text_field = pick_first_existing(columns, TEXT_FIELD_CANDIDATES)
    group_field = pick_first_existing(columns, GROUP_FIELD_CANDIDATES)

    if audio_field is None or text_field is None:
        raise ValueError(
            "Could not detect required columns. "
            f"Columns found: {columns}. "
            "Need at least one audio field and one text field."
        )

    if group_field is None:
        # We still allow run, but everything goes into a single group.
        group_field = "__all__"

    return FieldMap(audio=audio_field, text=text_field, group=group_field)


def safe_str(v):
    if v is None:
        return ""
    return str(v).strip()


def main():
    print("\n=== Accent Evaluation: Whisper + HF Dataset ===")
    print(f"Dataset: {DATASET_ID} | config={DATASET_CONFIG} | split={DATASET_SPLIT}")
    print(f"Model:   {ASR_MODEL_ID}")
    print(f"Max rows:{MAX_SAMPLES}\n")

    # 1) Load dataset
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    print(f"Loaded {len(ds)} rows")
    print(f"Columns: {ds.column_names}\n")

    # 2) Detect which columns to use
    fields = detect_fields(ds.column_names)
    print(f"Using fields -> audio: '{fields.audio}', text: '{fields.text}', group: '{fields.group}'")

    # 3) Ensure audio is decoded as array + sampling_rate
    ds = ds.cast_column(fields.audio, Audio())

    # 4) Initialize ASR pipeline
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL_ID)

    # 5) Iterate and compute WER by group
    group_scores = defaultdict(list)
    group_counts = defaultdict(int)

    processed = 0
    skipped = 0

    for idx, row in enumerate(ds):
        if processed >= MAX_SAMPLES:
            break

        ref = safe_str(row.get(fields.text))
        audio = row.get(fields.audio)

        # If no grouping field was found, collapse to one bucket.
        group_value = "all"
        if fields.group != "__all__":
            group_value = safe_str(row.get(fields.group)) or "unknown"

        # Validate row
        has_audio = isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio
        has_text = len(ref) > 0

        if STRICT_ROWS_ONLY and (not has_audio or not has_text):
            skipped += 1
            continue

        if not has_audio:
            skipped += 1
            continue

        # Run ASR
        try:
            hyp = asr({"array": audio["array"], "sampling_rate": audio["sampling_rate"]})["text"]
            hyp = safe_str(hyp)
            score = wer(ref.lower(), hyp.lower()) if has_text else math.nan

            if not math.isnan(score):
                group_scores[group_value].append(score)
                group_counts[group_value] += 1

            processed += 1
            print(f"[{processed:03d}] group={group_value:<15} WER={score:.3f}")

        except Exception as e:
            skipped += 1
            print(f"[skip] row={idx} group={group_value} error={e}")

    # 6) Print summary table
    print("\n=== Summary: Average WER by Group ===")
    if not group_scores:
        print("No valid scored rows. Check dataset fields or increase MAX_SAMPLES.")
    else:
        for group, scores in sorted(group_scores.items(), key=lambda x: (sum(x[1]) / len(x[1]))):
            avg_wer = sum(scores) / len(scores)
            print(f"{group:<20} avg_WER={avg_wer:.3f}  n={len(scores)}")

    print("\nRun stats")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")


if __name__ == "__main__":
    main()
