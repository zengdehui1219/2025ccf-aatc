import os
import librosa
import soundfile as sf
import numpy as np
import webrtcvad
from tqdm import tqdm
from shutil import move
from concurrent.futures import ThreadPoolExecutor, as_completed

vad = webrtcvad.Vad(2)  # 0: most strict, 3: most permissive

def is_corrupted(path):
    try:
        sf.read(path)
        return False
    except:
        return True

def vad_ratio(audio, sr, frame_duration_ms=30):
    audio = (audio * 32767).astype(np.int16)
    bytes_audio = audio.tobytes()
    frame_size = int(sr * frame_duration_ms / 1000) * 2  # 16-bit PCM

    total_frames = 0
    voiced_frames = 0
    for i in range(0, len(bytes_audio) - frame_size, frame_size):
        frame = bytes_audio[i:i + frame_size]
        if len(frame) < frame_size:
            continue
        total_frames += 1
        if vad.is_speech(frame, sr):
            voiced_frames += 1
    if total_frames == 0:
        return 0.0
    return voiced_frames / total_frames

def analyze_file(path, isolate_dir, min_duration=2.0, max_silence_ratio=0.75):
    try:
        if is_corrupted(path):
            move_to_isolate(path, isolate_dir, "Corrupted file")
            return (path, "Corrupted file")

        audio, sr = librosa.load(path, sr=44100, mono=True)
        duration = len(audio) / sr
        if duration < min_duration:
            move_to_isolate(path, isolate_dir, f"Too short: {duration:.2f}s")
            return (path, f"Too short: {duration:.2f}s")

        vad_r = vad_ratio(audio, sr)
        if vad_r > 0.1:
            move_to_isolate(path, isolate_dir, f"Contains voice: VAD ratio={vad_r:.2f}")
            return (path, f"Contains voice: VAD ratio={vad_r:.2f}")

        silence_ratio = 1.0 - vad_r
        if silence_ratio > max_silence_ratio:
            move_to_isolate(path, isolate_dir, f"Too much silence: {silence_ratio:.2f}")
            return (path, f"Too much silence: {silence_ratio:.2f}")

        return None  # Valid file
    except Exception as e:
        move_to_isolate(path, isolate_dir, f"Error: {str(e)}")
        return (path, f"Error: {str(e)}")

def move_to_isolate(path, isolate_dir, reason=""):
    os.makedirs(isolate_dir, exist_ok=True)
    dst = os.path.join(isolate_dir, os.path.basename(path))
    move(path, dst)

def scan_dataset_multithread(folder, isolate_dir, num_workers=8):
    audio_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    bad_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(analyze_file, path, isolate_dir): path for path in audio_files
        }
        for future in tqdm(as_completed(future_to_path), total=len(audio_files), desc="Scanning"):
            result = future.result()
            if result:
                bad_results.append(result)
    return bad_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/data2/noise/ESC-50-master/audio', help="Path to noise dataset")
    parser.add_argument("--isolate_dir", type=str, default="./isolated_noise", help="Directory to move bad files")
    parser.add_argument("--output_txt", type=str, default="bad_files.txt", help="Output list of bad files")
    parser.add_argument("--num_workers", type=int, default=24, help="Threads to use")
    args = parser.parse_args()

    bad_files = scan_dataset_multithread(args.data_dir, args.isolate_dir, args.num_workers)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for path, reason in bad_files:
            f.write(f"{path}|{reason}\n")

    print(f"\n✅ Done. Found {len(bad_files)} bad files. See: {args.output_txt}")
