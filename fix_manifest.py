import os
import json
import soundfile as sf
from datasets import load_from_disk

base = "/kaggle/working"
ds_path = f"{base}/tts_input_en_0p25h_ds"
out_dir = f"{base}/tts_outputs/en"
manifest = f"{out_dir}/manifest.jsonl"

os.makedirs(out_dir, exist_ok=True)
ds = load_from_disk(ds_path)

rows = []
for i, ex in enumerate(ds.select(range(min(20, len(ds))))):
    wav_out = os.path.abspath(os.path.join(out_dir, f"en_{i}.wav"))
    audio = ex["audio"]
    sf.write(wav_out, audio["array"], audio["sampling_rate"])
    rows.append({
        "wav": os.path.normpath(wav_out),
        "lang": "en",
        "ref_text": ex.get("text", ""),
        "tts_model_used": "dataset_audio"
    })

with open(manifest, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Rebuilt manifest:", manifest)
print("Rows:", len(rows))
print("Example wav:", rows[0]["wav"])
print("Exists:", os.path.isfile(rows[0]["wav"]))
