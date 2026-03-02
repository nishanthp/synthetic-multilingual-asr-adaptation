# Synthetic Multilingual ASR Adaptation

A research project exploring the use of synthetic speech generated through Text-to-Speech (TTS) systems for adapting and evaluating multilingual Automatic Speech Recognition (ASR) models.

## Overview

This repository investigates how synthetic multilingual speech can be leveraged to improve ASR systems across different languages, particularly for low-resource scenarios. By generating synthetic training data through TTS models, we aim to reduce the cost and time associated with collecting authentic transcribed audio while maintaining model performance.

## Key Features

- **Synthetic Speech Generation**: Generate multilingual synthetic speech using state-of-the-art TTS models
- **ASR Evaluation Pipeline**: Evaluate ASR model performance on synthetic vs. authentic speech
- **Multilingual Support**: Focus on English, Spanish, and French with extensibility to other languages
- **Closed-loop Analysis**: Study the TTS → ASR pipeline to understand synthetic speech characteristics

## Dataset

The synthetic multilingual speech dataset used in this project is available on Hugging Face:

  --> **[nprak26/synthetic-multilingual-speech-asr](https://huggingface.co/datasets/nprak26/synthetic-multilingual-speech-asr)**

### Dataset Specifications

- **Languages**: English (en), Spanish (es), French (fr)
- **Audio Format**: 16 kHz mono WAV files
- **Size**: Initial dataset with 15 examples (expandable)
- **License**: CC BY-NC 4.0

### Dataset Structure

Each sample contains:
- `wav`: Synthetic speech audio file
- `lang`: ISO language code
- `ref_text`: Reference text used for TTS generation
- `tts_model_used`: Identifier of the TTS model
- `hyp_text`: ASR system hypothesis (decoded transcript)

## Installation

```bash
# Clone the repository
git clone https://github.com/nishanthp/synthetic-multilingual-asr-adaptation.git
cd synthetic-multilingual-asr-adaptation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Speech

```python
from tts.multilingual_tts import generate_synthetic_audio

# Generate synthetic speech for a given text
audio = generate_synthetic_audio(
    text="Hello, this is a test.",
    language="en",
    tts_model="tts_models/multilingual/multi-dataset/your_tts"
)
```

### 2. Evaluate ASR Models

```python
from models.whisper_eval import evaluate_whisper
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("nprak26/synthetic-multilingual-speech-asr")

# Evaluate Whisper on synthetic speech
results = evaluate_whisper(dataset, model_size="base")
print(f"Word Error Rate: {results['wer']:.2%}")
```

### 3. Compare Synthetic vs. Authentic Speech

```python
from evaluation.compute_metrics import compare_speech_types

# Compare ASR performance on synthetic and authentic data
comparison = compare_speech_types(
    synthetic_dataset=dataset["synthetic"],
    authentic_dataset=dataset["authentic"]
)
```

## Research Applications

This project is designed for:

1. **Low-Resource ASR Development**: Bootstrap ASR systems for languages with limited transcribed audio
2. **Data Augmentation**: Supplement existing ASR training data with diverse synthetic examples
3. **Domain Adaptation**: Adapt pre-trained ASR models to specific accents or speaking styles
4. **Robustness Testing**: Evaluate ASR model generalization to synthetic speech patterns
5. **Cost-Effective Prototyping**: Rapidly prototype multilingual ASR systems without expensive data collection

## Methodology

### TTS Generation
- Utilize state-of-the-art multilingual TTS models (e.g., YourTTS, Coqui TTS)
- Generate diverse synthetic samples across multiple languages
- Control for speaker characteristics and prosody

### ASR Evaluation
- Test multiple ASR architectures (Whisper, Wav2Vec2.0, HuBERT)
- Compute standard metrics: Word Error Rate (WER), Character Error Rate (CER)
- Analyze performance degradation on synthetic vs. authentic speech

### Analysis
- Identify TTS characteristics that impact ASR performance
- Investigate optimal synthetic-to-authentic data ratios
- Study cross-lingual transfer learning potential

## Limitations

- **Scale**: Current dataset contains 15 samples; larger scale needed for production use
- **Diversity**: Synthetic speech lacks natural speaker variability, accents, and environmental noise
- **Domain**: Limited to clean, read speech; may not generalize to spontaneous conversation
- **Languages**: Currently supports only English, Spanish, and French

## Future Work

- [ ] Expand dataset to 1M+ synthetic utterances per language
- [ ] Add support for more languages (Mandarin, Hindi, Marathi, Kannada)
- [ ] Incorporate accent and dialect variation in TTS generation
- [ ] Experiment with mixed synthetic-authentic training strategies
- [ ] Develop speaker-adaptive synthetic generation techniques

## Citation

If you use this dataset or codebase in your research, please cite:

```bibtex
@misc{synthetic-multilingual-asr-2024,
  author = {Nishanth Prakash},
  title = {Synthetic Multilingual ASR Adaptation},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/nishanthp/synthetic-multilingual-asr-adaptation}},
  note = {Dataset: \url{https://huggingface.co/datasets/nprak26/synthetic-multilingual-speech-asr}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The dataset is available under the CC BY-NC 4.0 license.
