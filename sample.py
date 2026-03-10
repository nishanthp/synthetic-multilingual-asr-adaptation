import os
from datasets import load_dataset
from TTS.api import TTS

OUTPUT_DIR = "data/synthetic_audio"
NUM_SAMPLES = 10


def download_text_dataset():
    """
    Download an English-only dataset.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")

    # extract only valid sentences
    texts = [row["text"] for row in dataset if len(row["text"].strip()) > 50]

    return texts


def initialize_tts():
    """
    Load an open-source TTS model.
    """
    print("Loading TTS model...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    return tts


def generate_audio(tts, texts):
    """
    Convert text into synthetic speech.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, text in enumerate(texts[:NUM_SAMPLES]):

        file_path = os.path.join(OUTPUT_DIR, f"sample_{i}.wav")

        print(f"Generating speech for sample {i}")

        tts.tts_to_file(
            text=text,
            file_path=file_path
        )


def main():
    print("Downloading English dataset...")
    texts = download_text_dataset()

    print("Initializing TTS...")
    tts = initialize_tts()

    print("Generating synthetic speech...")
    generate_audio(tts, texts)

    print("Done! Synthetic speech files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()  
