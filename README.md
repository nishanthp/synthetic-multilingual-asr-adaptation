# Synthetic Multilingual ASR Adaptation

A research project exploring the use of synthetic speech generated through Text-to-Speech (TTS) systems for adapting and evaluating multilingual Automatic Speech Recognition (ASR) models.

## Overview

This repository investigates how synthetic multilingual speech can be used to improve ASR systems across languages, especially in low-resource settings. By generating synthetic training data with TTS, the goal is to reduce the cost and time of collecting transcribed audio while maintaining useful model performance.

## Key Features

- **Synthetic Speech Generation**: Generate multilingual synthetic speech with modern TTS models.
- **ASR Evaluation Pipeline**: Evaluate ASR performance on synthetic vs authentic speech.
- **Multilingual Support**: English, Spanish, and French (extensible to more languages).
- **Closed-Loop Analysis**: Study the TTS -> ASR pipeline and synthetic speech effects.

## Dataset

Hugging Face dataset: **[`nprak26/synthetic-multilingual-speech-asr`](https://huggingface.co/datasets/nprak26/synthetic-multilingual-speech-asr)**

### Dataset Specs

- **Languages**: English (`en`), Spanish (`es`), French (`fr`)
- **Audio format**: 16 kHz mono WAV
- **Current size**: 15 examples (expandable)
- **License**: CC BY-NC 4.0

### Dataset Fields

Each sample includes:

- `wav`: synthetic speech audio
- `lang`: ISO language code
- `ref_text`: reference text used for TTS generation
- `tts_model_used`: TTS model identifier
- `hyp_text`: ASR decoded transcript

## Installation

```bash
# Clone the repository
git clone https://github.com/nishanthp/synthetic-multilingual-asr-adaptation.git
cd synthetic-multilingual-asr-adaptation

# (Option A) Use uv
uv sync

# (Option B) Use venv + pip
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
pip install -r requirements.txt
```
Usage
1) Generate Synthetic Speech
python
```
from tts.multilingual_tts import generate_synthetic_audio

audio = generate_synthetic_audio(
    text="Hello, this is a test.",
    language="en",
    tts_model="tts_models/multilingual/multi-dataset/your_tts"
)
```
2) Evaluate ASR Models

```
from models.whisper_eval import evaluate_whisper
from datasets import load_dataset

dataset = load_dataset("nprak26/synthetic-multilingual-speech-asr")
results = evaluate_whisper(dataset, model_size="base")
print(f"Word Error Rate: {results['wer']:.2%}")
```
3) Compare Synthetic vs Authentic Speech

```
from evaluation.compute_metrics import compare_speech_types

comparison = compare_speech_types(
    synthetic_dataset=dataset["synthetic"],
    authentic_dataset=dataset["authentic"]
)
```

## Research Applications
Low-resource ASR bootstrapping
Data augmentation for existing ASR systems
Domain adaptation to accents/speaking styles
Robustness testing on synthetic speech patterns
Cost-effective multilingual ASR prototyping
Methodology
TTS Generation
Use multilingual TTS models (e.g., YourTTS, Coqui TTS)
Generate diverse synthetic samples across languages
Control speaker and prosody conditions where possible
ASR Evaluation
Evaluate Whisper, Wav2Vec2.0, HuBERT
Compute WER/CER
Analyze synthetic vs authentic performance differences
Analysis
Identify TTS artifacts that affect ASR quality
Evaluate synthetic-to-authentic data ratios
Study cross-lingual transfer potential
Limitations
Scale: Current dataset is small (15 samples)
Diversity: Synthetic speech lacks full real-world variability
Domain: Focused on clean/read speech
Languages: Currently English/Spanish/French only
Future Work
 Expand dataset to 1M+ synthetic utterances per language
 Add more languages (e.g., Mandarin, Hindi, Marathi, Kannada)
 Add accent/dialect variation in TTS generation
 Evaluate mixed synthetic-authentic training strategies
 Build speaker-adaptive synthetic generation approaches
Citation
bibtex

@misc{synthetic-multilingual-asr-2024,
  author = {Nishanth Prakash},
  title = {Synthetic Multilingual ASR Adaptation},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/nishanthp/synthetic-multilingual-asr-adaptation}},
  note = {Dataset: \url{https://huggingface.co/datasets/nprak26/synthetic-multilingual-speech-asr}}
}
Contributing
Contributions are welcome.

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m "Add AmazingFeature")
Push branch (git push origin feature/AmazingFeature)
Open a Pull Request
License
Add your project license here (for example, MIT, Apache-2.0, or CC BY-NC for dataset-specific components).