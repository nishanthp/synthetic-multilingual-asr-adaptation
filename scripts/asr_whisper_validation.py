# === SCRIPT 2: ASR VALIDATION (WHISPER, manual install) ===

import os, json, sys, subprocess

BASE = "/kaggle/working/tts_outputs/en"
MANIFEST = os.path.join(BASE, "manifest.jsonl")
METRICS_OUT = os.path.join(BASE, "metrics.json")

if not os.path.isfile(MANIFEST):
    raise SystemExit("❌ Manifest not found. Run Script 1 first.")

# --- deps ---
def ensure_whisper_from_git():
    try:
        import whisper
        return
    except ImportError:
        print("Installing whisper from GitHub (no pip torch downgrade)...")
        subprocess.run([
            "pip", "install", "git+https://github.com/openai/whisper.git"
        ], check=True)

ensure_whisper_from_git()
import whisper

# --- Metrics ---
def wer(ref: str, hyp: str) -> float:
    r = ref.strip().split(); h = hyp.strip().split()
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0]=i
    for j in range(len(h)+1): dp[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
    return dp[len(r)][len(h)]/max(1,len(r))

def cer(ref: str, hyp: str) -> float:
    r=list(ref.strip()); h=list(hyp.strip())
    dp=[[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0]=i
    for j in range(len(h)+1): dp[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
    return dp[len(r)][len(h)]/max(1,len(r))

# --- Load manifest safely ---
def load_manifest_safe(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠ Manifest is not a valid JSON array. Trying to read as JSONL...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

rows = load_manifest_safe(MANIFEST)
langs = sorted({r["lang"] for r in rows})
model = whisper.load_model("medium")  # CPU-friendly 2

metrics = {}
for lang in langs:
    items = [r for r in rows if r["lang"] == lang]
    wers, cers = [], []
    print(f"\n[{lang}] items={len(items)}")
    for r in items:
        wav = r["wav"]; ref = r["ref_text"]
        try:
            out = model.transcribe(wav, language=lang, fp16=False)
            hyp = (out.get("text") or "").strip()
            r["hyp"] = hyp
            r["WER"] = wer(ref.lower(), hyp.lower())
            r["CER"] = cer(ref.lower(), hyp.lower())
            wers.append(r["WER"]); cers.append(r["CER"])
            print(f"  {os.path.basename(wav)}  WER={r['WER']:.3f}  CER={r['CER']:.3f}  | “{hyp[:60]}”")
        except Exception as e:
            r["hyp"]=None; r["WER"]=None; r["CER"]=None
            print(f"  ASR failed for {wav}: {e}")

    metrics[lang] = {
        "clips": len([w for w in wers if w is not None]),
        "avg_WER": (sum(wers)/len(wers)) if wers else None,
        "avg_CER": (sum(cers)/len(cers)) if cers else None,
    }

# Save back results
with open(MANIFEST, "w", encoding="utf-8") as f: json.dump(rows, f, indent=2, ensure_ascii=False)
with open(METRICS_OUT, "w", encoding="utf-8") as f: json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\n================ METRICS ================")
for k, v in metrics.items():
    print(f"{k.upper()}: clips={v['clips']} | WER={None if v['avg_WER'] is None else round(v['avg_WER'],3)} | "
          f"CER={None if v['avg_CER'] is None else round(v['avg_CER'],3)}")
print(f"\nSaved: {METRICS_OUT}")

