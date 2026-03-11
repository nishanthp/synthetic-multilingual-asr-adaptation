# Works!!! === EN builder (LibriSpeech) — stable, no checkpoints ===
import os, time, shutil
import soundfile as sf
from datasets import load_dataset, Dataset, Audio

BASE_OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_en_audio")
MIN_FREE_MB  = 500
OVERWRITE    = True

# Targets / limits
TARGET_HOURS = 0.25      # ~15 minutes
MAX_MINUTES  = 20        # generous time budget
MAX_FILES    = 800       # safety cap

# Encoding: WAV is faster than FLAC (uses a bit more disk)
AUDIO_FMT    = "WAV"
WAV_SUBTYPE  = "PCM_16"

os.makedirs(BASE_OUT, exist_ok=True)

def free_mb(path=BASE_OUT):
    t,u,f = shutil.disk_usage(os.path.splitdrive(path)[0] or path)
    return f // (2**20)

def en_builder():
    tag       = str(TARGET_HOURS).replace(".","p")
    out_prefix= "tts_input_en"
    audio_dir = f"{BASE_OUT}/{out_prefix}_{tag}h_wav"
    ds_dir    = f"{BASE_OUT}/{out_prefix}_{tag}h_ds"

    if OVERWRITE:
        shutil.rmtree(ds_dir,    ignore_errors=True)
        # comment the next line if you want to append to existing audio
        shutil.rmtree(audio_dir, ignore_errors=True)

    os.makedirs(audio_dir, exist_ok=True)

    # Resume index from existing WAVs if any
    existing = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    start_i  = len(existing)

    rows = []
    saved = start_i
    total_sec = 0.0

    print(f"[start] free={free_mb()}MB; resume={start_i}; target={TARGET_HOURS}h")

    # LibriSpeech 'clean' config, 'train.100' split
    stream = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
    it = iter(stream)

    # If resuming, fast-forward through already “consumed” examples (one per saved file)
    for _ in range(start_i):
        try:
            next(it)
        except StopIteration:
            print("[end] nothing to resume; start at 0")
            break

    t0 = time.time()
    while True:
        # stop conditions
        if (time.time() - t0)/60.0 > MAX_MINUTES:
            print(f"[stop] time budget ~{(time.time()-t0)/60:.1f} min")
            break
        if (saved - start_i) >= MAX_FILES:
            print(f"[stop] file budget hit ({MAX_FILES})")
            break
        if free_mb() < MIN_FREE_MB:
            print(f"[stop] low space {free_mb()}MB")
            break

        try:
            ex = next(it)
        except StopIteration:
            print("[end] stream exhausted")
            break

        text = (ex.get("text") or "").strip()  # LibriSpeech uses 'text'
        if not text:
            continue

        a  = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        dur = len(a) / sr

        out_wav = os.path.join(audio_dir, f"en_{saved}.wav")
        sf.write(out_wav, a, sr, format=AUDIO_FMT, subtype=WAV_SUBTYPE)

        rows.append({"audio": out_wav, "text": text})
        total_sec += dur
        saved += 1

        if saved % 10 == 0:
            print(f"  files={saved}, hours={total_sec/3600:.2f}, free={free_mb()}MB")

        if total_sec/3600.0 >= TARGET_HOURS:
            print(f"[target] reached ~{TARGET_HOURS}h")
            break

    # Final save once (only if we actually collected rows)
    if not rows:
        print("[warn] no rows collected; dataset NOT saved")
        return None

    ds = Dataset.from_list(rows).cast_column("audio", Audio())
    os.makedirs(ds_dir, exist_ok=True)
    ds.save_to_disk(ds_dir)

    print(f"[done] saved={saved-start_i} new files, hours~{total_sec/3600:.2f}, ds={ds_dir}")
    return ds_dir

en_ds_dir = en_builder()
print("✅ EN subset at:", en_ds_dir)

