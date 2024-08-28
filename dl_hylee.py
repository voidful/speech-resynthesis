from datasets import load_dataset
import librosa
import soundfile as sf
import numpy as np
import os

dataset = load_dataset("voidful/gen_ai_2024")
output_folder = "gen_ai_2024_wav"
os.makedirs(output_folder, exist_ok=True)
target_sampling_rate = 16000

for i, sample in enumerate(dataset['train']):
    audio = sample['audio']['array']
    original_sampling_rate = sample['audio']['sampling_rate']
    if audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    audio_resampled = librosa.resample(audio, orig_sr=original_sampling_rate, target_sr=target_sampling_rate)
    output_path = os.path.join(output_folder, f"audio_{i}_16k.wav")
    sf.write(output_path, audio_resampled, target_sampling_rate)
    print(f"Saved {output_path}")

print("All audio files have been resampled to 16kHz and saved as WAV format.")
