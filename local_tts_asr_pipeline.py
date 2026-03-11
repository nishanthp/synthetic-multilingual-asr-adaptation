# === Local Windows Pipeline: TTS (edge-tts) -> ASR (faster-whisper) ===
# Replaces Coqui/XTTS step which requires Visual C++ build tools on Windows.
# Language: EN only (from downloaded English text)
# Outputs: tts_outputs/<lang>/manifest.jsonl + metrics.json

import asyncio
import json
import os
import re
import sys
import wave

BASE_OUT    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_OUT, "tts_outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_SAMPLES_PER_LANG = 100
TEXT_KEYS = ["text", "sentence", "transcription", "transcript", "normalized_text"]

EN_TEXT_JSONL = os.path.join(BASE_OUT, "data_en", "en_hf_text.jsonl")

# edge-tts voice per language code
VOICES = {
    "en": "en-US-JennyNeural",
}

FALLBACK_TEXTS = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we process speech.",
        "Hello, this is a multilingual text-to-speech test.",
        "The weather today is sunny with a light breeze.",
        "She opened the book and started reading from the first chapter.",
    ],
    "es": [
        "El zorro marrón rápido salta sobre el perro perezoso.",
        "La inteligencia artificial transforma el procesamiento del habla.",
        "Hola, esta es una prueba de síntesis de voz multilingüe.",
        "El tiempo hoy es soleado con una brisa suave.",
        "Ella abrió el libro y comenzó a leer desde el primer capítulo.",
    ],
    "fr": [
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "L'intelligence artificielle transforme le traitement de la parole.",
        "Bonjour, ceci est un test de synthèse vocale multilingue.",
        "Le temps aujourd'hui est ensoleillé avec une légère brise.",
        "Elle a ouvert le livre et a commencé à lire le premier chapitre.",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    return t[:220] + "..." if len(t) > 220 else t


def load_en_texts_from_dataset(ds_dir: str, k: int) -> list:
    try:
        from datasets import load_from_disk
        ds = load_from_disk(ds_dir)
        print(f"  ✓ HF dataset: {ds_dir} | rows={len(ds)}")
        texts = []
        for i in range(min(k * 4, len(ds))):
            ex = ds[i]
            for key in TEXT_KEYS:
                v = ex.get(key)
                if isinstance(v, str) and v.strip():
                    texts.append(clean_text(v.strip()))
                    break
            if len(texts) >= k:
                break
        return texts
    except Exception as e:
        print(f"  ⚠ Could not load dataset from {ds_dir}: {e}")
        return []


def load_en_texts_from_jsonl(jsonl_path: str, k: int) -> list:
    texts = []
    if not os.path.isfile(jsonl_path):
        print(f"  ⚠ English text jsonl not found: {jsonl_path}")
        return texts
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                txt = obj.get("text") if isinstance(obj, dict) else None
                if isinstance(txt, str) and txt.strip():
                    texts.append(clean_text(txt.strip()))
                if len(texts) >= k:
                    break
    except Exception as e:
        print(f"  ⚠ Could not load english text jsonl {jsonl_path}: {e}")
    return texts


# ---------------------------------------------------------------------------
# TTS (edge-tts, async)
# ---------------------------------------------------------------------------
async def synthesise_one(text: str, out_wav: str, voice: str):
    import av
    import edge_tts
    import numpy as np

    def convert_mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int = 16000):
        container = av.open(mp3_path)
        if not container.streams.audio:
            raise RuntimeError(f"No audio stream found in {mp3_path}")

        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks = []

        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            frames = resampled if isinstance(resampled, list) else [resampled]
            for rf in frames:
                arr = rf.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr.astype(np.int16).tobytes())

        container.close()

        if not chunks:
            raise RuntimeError(f"Decoded empty audio while converting {mp3_path}")

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(chunks))

    communicate = edge_tts.Communicate(text, voice)
    # edge-tts outputs MP3; we convert to PCM WAV and keep WAV only
    mp3_path = out_wav.replace(".wav", ".mp3")
    await communicate.save(mp3_path)
    try:
        convert_mp3_to_wav(mp3_path, out_wav, sample_rate=16000)
        os.remove(mp3_path)
        return out_wav
    except Exception as e:
        if os.path.isfile(mp3_path):
            os.remove(mp3_path)
        raise RuntimeError(f"WAV conversion failed for {out_wav}: {e}")


async def synthesise_all(lang_texts: dict, lang_dirs: dict) -> dict:
    """Returns {lang: [(wav_path, ref_text), ...]}"""
    results = {}
    for lang, texts in lang_texts.items():
        voice = VOICES[lang]
        out_dir = lang_dirs[lang]
        pairs = []
        print(f"\n[TTS] {lang.upper()} — voice={voice}, sentences={len(texts)}")
        for i, text in enumerate(texts):
            out_wav = os.path.join(out_dir, f"{lang}_{i}.wav")
            try:
                final_path = await synthesise_one(text, out_wav, voice)
                if os.path.isfile(final_path) and os.path.getsize(final_path) > 0:
                    pairs.append((final_path, text))
                    print(f"  ✓ {os.path.basename(final_path)}")
                else:
                    print(f"  ✗ empty output for: {text[:50]}")
            except Exception as e:
                print(f"  ✗ TTS failed [{lang}] clip {i}: {e}")
        results[lang] = pairs
    return results


# ---------------------------------------------------------------------------
# ASR + WER/CER (faster-whisper)
# ---------------------------------------------------------------------------
def _lev(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    if not r and not h:
        return 0.0
    return _lev(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref.lower()), list(hyp.lower())
    if not r and not h:
        return 0.0
    return _lev(r, h) / max(1, len(r))


def run_asr(lang_results: dict, lang_dirs: dict) -> dict:
    from faster_whisper import WhisperModel
    print("\n=== ASR Validation (faster-whisper base, CPU, int8) ===")
    asr = WhisperModel("base", device="cpu", compute_type="int8")
    metrics = {}

    for lang, pairs in lang_results.items():
        out_dir = lang_dirs[lang]
        manifest_path = os.path.join(out_dir, "manifest.jsonl")
        wers, cers = [], []
        rows = []

        for wav_path, ref_text in pairs:
            hyp = ""
            try:
                segs, _ = asr.transcribe(wav_path, beam_size=1, language=lang)
                hyp = " ".join(s.text.strip() for s in segs).strip()
            except Exception as e:
                print(f"  ✗ ASR failed [{lang}] {os.path.basename(wav_path)}: {e}")

            w = wer(ref_text, hyp)
            c = cer(ref_text, hyp)
            wers.append(w)
            cers.append(c)
            rows.append({
                "wav": wav_path,
                "lang": lang,
                "ref_text": ref_text,
                "hyp_text": hyp,
                "WER": round(w, 4),
                "CER": round(c, 4),
                "tts_model": f"edge-tts/{VOICES[lang]}",
            })
            print(f"  [{lang}] {os.path.basename(wav_path)}  WER={w:.3f}  CER={c:.3f}  | \"{hyp[:60]}\"")

        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        avg_wer = sum(wers) / len(wers) if wers else None
        avg_cer = sum(cers) / len(cers) if cers else None
        metrics[lang] = {"clips": len(rows), "avg_WER": avg_wer, "avg_CER": avg_cer}
        print(f"  → {lang.upper()}: clips={len(rows)} | avg_WER={round(avg_wer,3) if avg_wer is not None else None} | avg_CER={round(avg_cer,3) if avg_cer is not None else None}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # 1. Collect texts
    lang_texts = {}
    for lang in VOICES:
        texts = load_en_texts_from_jsonl(EN_TEXT_JSONL, MAX_SAMPLES_PER_LANG) if lang == "en" else []
        if not texts:
            print(f"  ⚠ No dataset texts for {lang.upper()}, using fallback sentences.")
            texts = FALLBACK_TEXTS.get(lang, [])[:MAX_SAMPLES_PER_LANG]
        lang_texts[lang] = texts[:MAX_SAMPLES_PER_LANG]
        print(f"  [{lang.upper()}] using {len(lang_texts[lang])} sentences")

    # 2. Create output dirs
    lang_dirs = {}
    for lang in VOICES:
        d = os.path.join(OUTPUT_ROOT, lang)
        os.makedirs(d, exist_ok=True)
        lang_dirs[lang] = d

    # 3. TTS synthesis
    lang_results = await synthesise_all(lang_texts, lang_dirs)

    # 4. ASR validation
    metrics = run_asr(lang_results, lang_dirs)

    # 5. Save summary metrics
    metrics_path = os.path.join(OUTPUT_ROOT, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 55)
    for lang, m in metrics.items():
        print(f"  {lang.upper()}: clips={m['clips']} | WER={m.get('avg_WER')} | CER={m.get('avg_CER')}")
    print(f"\n  Manifests + metrics saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    asyncio.run(main())
