import os
import wave

audio_folder = "gen_ai_2024_wav"
manifest_file = "manifest.txt"
root_directory = os.path.abspath(audio_folder)
with open(manifest_file, "w") as manifest:
    manifest.write(f"{root_directory}\n")
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(audio_folder, audio_file)
            with wave.open(audio_path, "rb") as wav_file:
                num_frames = wav_file.getnframes()
            relative_path = os.path.relpath(audio_path, root_directory)
            manifest.write(f"{relative_path}\t{num_frames}\n")

print(f"MANIFEST file created successfully at {manifest_file}")
