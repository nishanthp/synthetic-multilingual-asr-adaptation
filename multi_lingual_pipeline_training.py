# === Multilingual pipeline: Build -> TTS (XTTS) -> ASR (optional) ===
# Languages: EN, ES, FR
# Outputs:
#   TTS wavs: /kaggle/working/tts_outputs/<lang>/*.wav
#   TTS manifest: /kaggle/working/tts_outputs/<lang>/manifest.jsonl
#   (Optional) WER/CER summary

import os, re, time, json, glob, random, subprocess, sys
from typing import List, Optional, Dict

# ------------------ TOGGLES ------------------
RUN_BUILDERS = False      # True to (re)build small EN/ES/FR subsets
RUN_TTS      = True
RUN_ASR      = True       # requires faster-whisper; auto-skips if not available

# ------------------ PATHS & CONFIG ------------------
# BASE_OUT = "/kaggle/working"
BASE_OUT = "./lab_output"

OUTPUT_ROOT = f"{BASE_OUT}/tts_outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Saved dataset dirs we will USE (created by builders or already present)
DATASET_DIRS: Dict[str, List[str]] = {
    "en": [f"{BASE_OUT}/tts_input_en_0p25h_ds", f"{BASE_OUT}/mls_en_0p1h_ds"],
    "es": [f"{BASE_OUT}/mls_es_0p1h_ds"],
    "fr": [f"{BASE_OUT}/mls_fr_0p1h_ds"],
}

LANGS = ["en", "es", "fr"]
MAX_SAMPLES_PER_LANG = 5
TEXT_KEYS = ["text", "sentence", "transcription", "transcript", "normalized_text"]
random.seed(13)

# Fallback tiny single-language models (also used to bootstrap a speaker reference if none found)
BOOTSTRAP_MODELS = {
    "en": ("tts_models/en/ljspeech/tacotron2-DDC", "Hello, this is a short reference."),
    "es": ("tts_models/es/css10/vits", "Hola, esta es una referencia corta."),
    "fr": ("tts_models/fr/css10/vits", "Bonjour, ceci est une courte référence."),
}

# ------------------ UTILS ------------------
def ensure_pkg(mod_name: str, pip_name: Optional[str] = None, version: Optional[str] = None):
    try:
        __import__(mod_name)
        return True
    except Exception:
        pkg = pip_name or mod_name
        if version:
            pkg = f"{pkg}=={version}"
        print(f"🔧 Installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", pkg], check=True)
        try:
            __import__(mod_name)
            return True
        except Exception:
            return False

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    return t[:220] + "..." if len(t) > 220 else t

def find_hf_ds(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isdir(p) and \
           os.path.isfile(os.path.join(p, "dataset_info.json")) and \
           os.path.isdir(os.path.join(p, "data")):
                return p
    return None

# ------------------ (A) BUILDERS (optional) ------------------
if RUN_BUILDERS:
    import soundfile as sf
    from datasets import load_dataset, Dataset, Audio
    MIN_FREE_MB  = 1500
    TARGET_HOURS = 0.25
    MAX_MINUTES  = 20
    AUDIO_FMT, WAV_SUBTYPE = "WAV", "PCM_16"
    OVERWRITE = True

    def free_mb(path="/kaggle"):
        t,u,f = shutil.disk_usage(path)
        return f // (2**20)

    import shutil

    # EN: LibriSpeech clean train.100 (simple, no checkpoints)
    def build_en():
        tag = str(TARGET_HOURS).replace(".","p")
        audio_dir = f"{BASE_OUT}/tts_input_en_{tag}h_wav"
        ds_dir    = f"{BASE_OUT}/tts_input_en_{tag}h_ds"
        if OVERWRITE:
            shutil.rmtree(ds_dir, ignore_errors=True)
            shutil.rmtree(audio_dir, ignore_errors=True)
        os.makedirs(audio_dir, exist_ok=True)

        rows, total_sec, saved = [], 0.0, 0
        stream = iter(load_dataset("librispeech_asr", "clean", split="train.100", streaming=True))
        t0 = time.time()
        while True:
            if (time.time()-t0)/60.0 > MAX_MINUTES or free_mb()<MIN_FREE_MB:
                break
            ex = next(stream, None)
            if ex is None:
                break
            text = (ex.get("text") or "").strip()
            if not text: continue
            a  = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
            dur = len(a)/sr
            out_wav = os.path.join(audio_dir, f"en_{saved}.wav")
            sf.write(out_wav, a, sr, format=AUDIO_FMT, subtype=WAV_SUBTYPE)
            rows.append({"audio": out_wav, "text": text})
            total_sec += dur; saved += 1
            if total_sec/3600.0 >= TARGET_HOURS: break
        if rows:
            ds = Dataset.from_list(rows).cast_column("audio", Audio()); os.makedirs(ds_dir, exist_ok=True); ds.save_to_disk(ds_dir)
            print("[EN] saved:", ds_dir)
        return ds_dir if rows else None

    # ES/FR: MLS streaming (fast, with stride)
    def build_mls_lang(config:str, lang_tag:str, out_prefix:str):
        STRIDE, PRINT_EVERY, SAVE_EVERY = 3, 10, 50
        tag = str(TARGET_HOURS).replace(".","p")
        audio_dir = f"{BASE_OUT}/{out_prefix}_{tag}h_wav"
        ds_dir    = f"{BASE_OUT}/{out_prefix}_{tag}h_ds"
        if OVERWRITE:
            shutil.rmtree(ds_dir, ignore_errors=True)
            shutil.rmtree(audio_dir, ignore_errors=True)
        os.makedirs(audio_dir, exist_ok=True); os.makedirs(ds_dir, exist_ok=True)

        existing = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")]); start_i = len(existing)
        rows, total_sec, saved, processed_valid = [], 0.0, start_i, 0
        stream = iter(load_dataset("facebook/multilingual_librispeech", config, split="train", streaming=True))

        # fast-forward for resume alignment
        for _ in range(start_i * STRIDE):
            next(stream, None)

        t0 = time.time()
        def get_text_any(ex):
            for k in ("text","sentence","transcript","transcription","normalized_text"):
                v = ex.get(k); 
                if isinstance(v,str) and v.strip(): return v.strip()
            return ""
        while True:
            if (time.time()-t0)/60.0 > MAX_MINUTES or free_mb()<MIN_FREE_MB: break
            ex = next(stream, None)
            if ex is None: break
            text = get_text_any(ex)
            if not text: continue
            if processed_valid % STRIDE != 0:
                processed_valid += 1; continue
            processed_valid += 1
            try:
                a  = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
                dur = len(a)/sr
                out_wav = os.path.join(audio_dir, f"{lang_tag}_{saved}.wav")
                sf.write(out_wav, a, sr, format=AUDIO_FMT, subtype=WAV_SUBTYPE)
                rows.append({"audio": out_wav, "text": text})
                total_sec += dur; saved += 1
                if saved % PRINT_EVERY == 0:
                    print(f"[{lang_tag}] files={saved}, hours={total_sec/3600:.2f}")
                if (saved - start_i) % SAVE_EVERY == 0 and rows:
                    Dataset.from_list(rows).cast_column("audio", Audio()).save_to_disk(ds_dir)
                if total_sec/3600.0 >= TARGET_HOURS: break
            except Exception as e:
                if saved - start_i < 3: print(f"[{lang_tag}] skip early:", e)
                continue
        if rows:
            Dataset.from_list(rows).cast_column("audio", Audio()).save_to_disk(ds_dir)
            print(f"[{lang_tag}] saved:", ds_dir)
        return ds_dir if rows else None

    print("🏗️ Building small subsets (may skip if already present)...")
    try:
        build_mls_lang("spanish", "es", "mls_es")   # ES
    except Exception as e:
        print("ES builder failed:", e)
    try:
        build_mls_lang("french", "fr", "mls_fr")    # FR
    except Exception as e:
        print("FR builder failed:", e)
    try:
        build_en()                                   # EN
    except Exception as e:
        print("EN builder failed:", e)

# ------------------ (B) TTS with XTTS (speaker refs) ------------------
if RUN_TTS:
    ok_datasets = ensure_pkg("datasets")
    ok_tts = ensure_pkg("TTS")  # as-is (no extra pins)
    if not (ok_datasets and ok_tts):
        raise RuntimeError("Missing required packages: datasets or TTS.")

    from datasets import load_from_disk
    from TTS.api import TTS

    def load_texts(ds_path: str, k: int) -> List[str]:
        ds = load_from_disk(ds_path)
        print(f"  ✓ HF dataset: {ds_path} | rows={len(ds)} | cols={getattr(ds, 'column_names', [])}")
        out = []
        scan = min(k * 6, len(ds))
        for i in range(scan):
            ex = ds[i]; txt = ""
            if isinstance(ex, dict):
                for key in TEXT_KEYS:
                    v = ex.get(key)
                    if isinstance(v, str) and v.strip():
                        txt = v.strip(); break
            if txt:
                out.append(clean_text(txt))
            if len(out) >= k: break
        if not out: print("  ⚠ No usable texts; will use fallback.")
        return out

    def fallback_texts(lang: str) -> List[str]:
        samples = {
            "en": ["Hello, this is a multilingual TTS test.",
                   "The quick brown fox jumps over the lazy dog.",
                   "We generate audio from text automatically.",
                   "Streaming datasets help rapid prototyping.",
                   "Final English sample sentence."],
            "es": ["Hola, esta es una prueba de TTS multilingüe.",
                   "El zorro marrón rápido salta sobre el perro perezoso.",
                   "Generamos audio a partir de texto automáticamente.",
                   "Los conjuntos de datos en streaming ayudan a prototipar.",
                   "Última frase de ejemplo en español."],
            "fr": ["Bonjour, ceci est un test de synthèse vocale multilingue.",
                   "Le vif renard brun saute par-dessus le chien paresseux.",
                   "Nous générons de l'audio à partir du texte automatiquement.",
                   "Les jeux de données en streaming aident au prototypage.",
                   "Dernière phrase d'exemple en français."]
        }
        return [clean_text(x) for x in samples.get(lang, samples["en"])]

    def find_speaker_wav(ds_path: Optional[str]) -> Optional[str]:
        if ds_path:
            try:
                ds = load_from_disk(ds_path)
                scan = min(200, len(ds))
                for i in range(scan):
                    ex = ds[i]
                    if isinstance(ex, dict) and "audio" in ex:
                        aud = ex["audio"]
                        if isinstance(aud, dict):
                            p = aud.get("path")
                            if p and os.path.isfile(p) and os.path.getsize(p) > 0:
                                return p
            except Exception as e:
                print(f"  ⚠ Could not read audio paths from {ds_path}: {e}")
            # glob as fallback
            wavs = glob.glob(os.path.join(ds_path, "**", "*.wav"), recursive=True)
            for w in wavs:
                if os.path.getsize(w) > 0: return w
        return None

    def bootstrap_speaker(lang: str, tmp_dir: str) -> Optional[str]:
        model_id, ref_text = BOOTSTRAP_MODELS[lang]
        os.makedirs(tmp_dir, exist_ok=True)
        out = os.path.join(tmp_dir, f"{lang}_ref.wav")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        try:
            print(f"  🔧 Bootstrapping speaker for {lang.upper()} using {model_id}")
            local = TTS(model_name=model_id, gpu=False)
            local.tts_to_file(text=ref_text, file_path=out)
            return out if os.path.isfile(out) and os.path.getsize(out) > 0 else None
        except Exception as e:
            print(f"  ⚠ Bootstrap failed for {lang.upper()} (non-fatal): {e}")
            return None

    # --- Load multilingual TTS (XTTS -> YourTTS fallback) and detect supported langs
    print("\n=== Loading multilingual TTS (XTTS v2 → YourTTS fallback) ===")
    tts, model_name = None, None
    for mn, desc in [("tts_models/multilingual/multi-dataset/xtts_v2","XTTS v2"),
                     ("tts_models/multilingual/multi-dataset/your_tts","YourTTS")]:
        try:
            print(f"  Attempt: {desc} ({mn})")
            tts = TTS(model_name=mn, gpu=False)
            model_name = mn; print(f"  ✅ Using {desc}")
            break
        except Exception as e:
            print(f"  ✗ {desc} failed: {e}")
    if not tts:
        raise RuntimeError("Could not load XTTS or YourTTS.")

    def _get_available_langs(tts_obj) -> set:
        # Try to introspect supported language codes from TTS internals
        try:
            syn = getattr(tts_obj, "synthesizer", None)
            tm  = getattr(syn, "tts_model", None)
            lm  = getattr(tm, "language_manager", None)
            if lm and hasattr(lm, "lang_ids") and isinstance(lm.lang_ids, dict):
                return set(lm.lang_ids.keys())
        except Exception:
            pass
        # Fallback: known YourTTS set
        return set(["en", "fr-fr", "pt-br"])

    AVAILABLE_LANGS = _get_available_langs(tts)
    print("Active TTS model:", model_name, "| langs:", sorted(list(AVAILABLE_LANGS)))

    # STEP 1: collect texts + speaker per language
    lang_texts: Dict[str, List[str]] = {}
    lang_ds_dir: Dict[str, Optional[str]] = {}
    for lang in LANGS:
        ds_dir = find_hf_ds(DATASET_DIRS.get(lang, []))
        lang_ds_dir[lang] = ds_dir
        if ds_dir:
            texts = load_texts(ds_dir, MAX_SAMPLES_PER_LANG)
        else:
            print(f"⚠ No dataset dir for {lang.upper()}; using fallback.")
            texts = []
        if not texts: texts = fallback_texts(lang)[:MAX_SAMPLES_PER_LANG]
        lang_texts[lang] = texts

    tmp_ref_dir = f"{BASE_OUT}/_xtts_refs"; os.makedirs(tmp_ref_dir, exist_ok=True)
    lang_speaker: Dict[str, Optional[str]] = {}
    for lang in LANGS:
        sp = find_speaker_wav(lang_ds_dir[lang]) or bootstrap_speaker(lang, tmp_ref_dir)
        if sp: print(f"🎙️ {lang.upper()} speaker_wav: {sp}")
        else:  print(f"⚠ No speaker_wav for {lang.upper()} (will proceed without; quality may vary).")
        lang_speaker[lang] = sp

    # ---- Fallback-aware synth helper
    def synth_with_fallback(text: str, wav_path: str, lang: str, speaker_wav: Optional[str]) -> str:
        """
        Return the model_id actually used for synthesis.
        """
        if lang in AVAILABLE_LANGS:
            # Use active multilingual model directly
            if speaker_wav:
                tts.tts_to_file(text=text, file_path=wav_path, language=lang, speaker_wav=speaker_wav)
            else:
                tts.tts_to_file(text=text, file_path=wav_path, language=lang)
            return model_name

        # Multilingual model doesn't support this language → single-lang fallback
        if lang in BOOTSTRAP_MODELS:
            single_model_id, _ = BOOTSTRAP_MODELS[lang]
            print(f"  ↪ Fallback to single-lang model for '{lang}': {single_model_id}")
            local = TTS(model_name=single_model_id, gpu=False)
            local.tts_to_file(text=text, file_path=wav_path)
            return single_model_id

        raise RuntimeError(f"No TTS available for language '{lang}' in current model ({model_name}) and no fallback model configured.")

    # STEP 2: synthesize
    manifests = {}
    for lang in LANGS:
        out_dir = os.path.join(OUTPUT_ROOT, lang); os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(out_dir, "manifest.jsonl")
        generated = 0
        with open(manifest_path, "w", encoding="utf-8") as mf:
            for i, text in enumerate(lang_texts[lang]):
                wav_path = os.path.join(out_dir, f"{lang}_{i}.wav")
                try:
                    used_model = synth_with_fallback(text, wav_path, lang, lang_speaker[lang])
                    if os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
                        generated += 1
                        mf.write(json.dumps({"wav": wav_path,
                                             "lang": lang,
                                             "ref_text": text,
                                             "tts_model_used": used_model}, ensure_ascii=False) + "\n")
                        print(f"  ✓ [{lang}] {os.path.basename(wav_path)}  ({used_model})")
                    else:
                        print(f"  ✗ [{lang}] empty output: {wav_path}")
                except Exception as e:
                    print(f"  ✗ [{lang}] synth failed ({os.path.basename(wav_path)}): {e}")
        manifests[lang] = {"dir": out_dir, "manifest": manifest_path, "generated": generated}
        print(f"→ {lang.upper()}: generated={generated} wavs → {out_dir}")
    print(f"\n🤖 TTS model primary: {model_name}")

# ------------------ (C) ASR + WER/CER (optional) ------------------
if RUN_ASR:
    ok_asr = ensure_pkg("faster_whisper", "faster-whisper")
    if not ok_asr:
        print("⚠ faster-whisper not available; skipping ASR.")
    else:
        from faster_whisper import WhisperModel

        def _lev(a: List[str], b: List[str]) -> int:
            m, n = len(a), len(b)
            dp = list(range(n+1))
            for i in range(1, m+1):
                prev, dp[0] = dp[0], i
                for j in range(1, n+1):
                    cur = dp[j]
                    dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (a[i-1]!=b[j-1]))
                    prev = cur
            return dp[n]
        def wer(ref: str, hyp: str) -> float:
            r, h = ref.lower().split(), hyp.lower().split()
            return 0.0 if not r and not h else (_lev(r,h) / max(1,len(r)))
        def cer(ref: str, hyp: str) -> float:
            r, h = list(ref.lower()), list(hyp.lower())
            return 0.0 if not r and not h else (_lev(r,h) / max(1,len(r)))

        print("\n=== Transcribing with faster-whisper (base, CPU, int8) ===")
        asr = WhisperModel("base", device="cpu", compute_type="int8")
        results = {}
        for lang in LANGS:
            out_dir = os.path.join(OUTPUT_ROOT, lang)
            manifest = os.path.join(out_dir, "manifest.jsonl")
            if not os.path.isfile(manifest):
                results[lang] = {"num":0, "wer":None, "cer":None}
                continue
            with open(manifest, "r", encoding="utf-8") as f:
                rows = [json.loads(x) for x in f if x.strip()]
            total_w, total_c, cnt = 0.0, 0.0, 0
            enriched = []
            for row in rows:
                wav, ref = row["wav"], row["ref_text"]
                hyp = ""
                try:
                    segs, info = asr.transcribe(wav, beam_size=1, language=lang)
                    hyp = " ".join(s.text.strip() for s in segs).strip()
                except Exception as e:
                    print(f"  ✗ [{lang}] ASR failed for {os.path.basename(wav)}: {e}")
                row["hyp_text"] = hyp
                enriched.append(row)
                if ref and hyp:
                    total_w += wer(ref,hyp); total_c += cer(ref,hyp); cnt += 1
            with open(manifest, "w", encoding="utf-8") as f:
                for r in enriched:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            results[lang] = {"num": len(enriched), "wer": (total_w/cnt if cnt else None), "cer": (total_c/cnt if cnt else None)}

        # --- Final report
        print("\n" + "="*60)
        print("PIPELINE DONE — SUMMARY")
        print("="*60)
        for lang in LANGS:
            m = manifests[lang] if RUN_TTS else {"dir": os.path.join(OUTPUT_ROOT, lang), "generated": None}
            r = results.get(lang, {})
            print(f"{lang.upper()}: wavs={m.get('generated')} | items={r.get('num')} | WER={r.get('wer')} | CER={r.get('cer')} | dir={m.get('dir')}")
        print(f"\n📁 Outputs root: {OUTPUT_ROOT}")

