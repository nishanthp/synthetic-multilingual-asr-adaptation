# Works!!! === FR builder (MLS) — fast, with dynamic text key detection ===
import os, time, shutil
import soundfile as sf
from datasets import load_dataset, Dataset, Audio

BASE_OUT     = "/kaggle/working"
MIN_FREE_MB  = 1500
OVERWRITE    = True

TARGET_HOURS = 0.10
MAX_MINUTES  = 15
MAX_FILES    = 220
STRIDE       = 3
PRINT_EVERY  = 10
SAVE_EVERY   = 40

AUDIO_FMT    = "WAV"
WAV_SUBTYPE  = "PCM_16"

os.makedirs(BASE_OUT, exist_ok=True)

def free_mb(path="/kaggle"):
    t,u,f = shutil.disk_usage(path)
    return f // (2**20)

def detect_text_key(example):
    for k in ["transcription", "normalized_text", "transcript", "text"]:
        if k in example and isinstance(example[k], str) and example[k].strip():
            return k
    return None

def fr_builder():
    tag        = str(TARGET_HOURS).replace(".","p")
    out_prefix = "mls_fr"
    audio_dir  = f"{BASE_OUT}/{out_prefix}_{tag}h_wav"
    ds_dir     = f"{BASE_OUT}/{out_prefix}_{tag}h_ds"
    tmp_ds     = os.path.join(BASE_OUT, "_partial_ds_tmp_fr")

    if OVERWRITE:
        shutil.rmtree(ds_dir, ignore_errors=True)
        shutil.rmtree(audio_dir, ignore_errors=True)

    os.makedirs(audio_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    start_i  = len(existing)
    rows = []
    saved = start_i
    total_sec = 0.0

    print(f"[start FR] free={free_mb()}MB; resume index={start_i}; target={TARGET_HOURS}h")
    stream = iter(load_dataset("facebook/multilingual_librispeech", "french", split="train", streaming=True))

    # Detect which key to use for text
    first_ex = next(stream)
    text_key = detect_text_key(first_ex)
    if not text_key:
        raise ValueError("No usable text key found in dataset.")
    print(f"[info FR] Using text field: {text_key}")

    # process first sample if stride allows and we are not resuming
    if start_i == 0 and 0 % STRIDE == 0:
        if first_ex[text_key].strip():
            a = first_ex["audio"]["array"]
            sr = first_ex["audio"]["sampling_rate"]
            dur = len(a) / sr
            out_wav = os.path.join(audio_dir, f"fr_{saved}.wav")
            sf.write(out_wav, a, sr, format=AUDIO_FMT, subtype=WAV_SUBTYPE)
            rows.append({"audio": out_wav, "text": first_ex[text_key].strip()})
            total_sec += dur
            saved += 1

    t0 = time.time()
    for raw_idx, ex in enumerate(stream, start=1):
        # stride down the stream for speed
        if raw_idx % STRIDE != 0:
            continue

        # stop conditions
        if (time.time() - t0)/60.0 > MAX_MINUTES:
            print(f"[stop FR] time budget ~{(time.time()-t0)/60:.1f} min")
            break
        if saved - start_i >= MAX_FILES:
            print(f"[stop FR] file budget hit ({MAX_FILES})")
            break
        if free_mb() < MIN_FREE_MB:
            print(f"[stop FR] low space {free_mb()}MB")
            break

        text = (ex.get(text_key) or "").strip()
        if not text:
            continue

        a  = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        dur = len(a) / sr

        out_wav = os.path.join(audio_dir, f"fr_{saved}.wav")
        sf.write(out_wav, a, sr, format=AUDIO_FMT, subtype=WAV_SUBTYPE)

        rows.append({"audio": out_wav, "text": text})
        total_sec += dur
        saved += 1

        if saved % PRINT_EVERY == 0:
            print(f"  [FR] files={saved}, hours={total_sec/3600:.2f}, free={free_mb()}MB")

        if (saved - start_i) % SAVE_EVERY == 0 and rows:
            Dataset.from_list(rows).cast_column("audio", Audio()).save_to_disk(tmp_ds)
            shutil.rmtree(ds_dir, ignore_errors=True)
            shutil.move(tmp_ds, ds_dir)

        if total_sec/3600.0 >= TARGET_HOURS:
            print(f"[target FR] reached ~{TARGET_HOURS}h")
            break

    # final save if we collected anything
    if rows:
        Dataset.from_list(rows).cast_column("audio", Audio()).save_to_disk(tmp_ds)
        shutil.rmtree(ds_dir, ignore_errors=True)
        shutil.move(tmp_ds, ds_dir)
        print(f"[done FR] saved={saved}, hours~{total_sec/3600:.2f}, ds={ds_dir}")
    else:
        print("[warn FR] No rows collected, nothing saved.")

    return ds_dir if rows else None

fr_ds_dir = fr_builder()
print("✅ FR subset at:", fr_ds_dir)

