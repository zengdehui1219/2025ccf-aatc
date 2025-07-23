import os
import librosa
import soundfile as sf
import numpy as np
import shutil
import webrtcvad
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

vad = webrtcvad.Vad(3)  # 0: strict, 3: tolerant

def is_corrupted(path):
    try:
        sf.read(path)
        return False
    except:
        return True

def is_voice_present(audio, sr, frame_duration_ms=30):
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = (audio * 32767).astype(np.int16)
    bytes_audio = audio.tobytes()
    frame_size = int(sr * frame_duration_ms / 1000) * 2  # 16-bit PCM
    for i in range(0, len(bytes_audio) - frame_size, frame_size):
        frame = bytes_audio[i:i + frame_size]
        if len(frame) < frame_size:
            continue
        if vad.is_speech(frame, sr):
            return True
    return False

def silent_ratio(audio, frame_length=1024, hop_length=512, threshold=1e-4):
    total_frames = 0
    silent_frames = 0
    for i in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        total_frames += 1
        if rms < threshold:
            silent_frames += 1
    if total_frames == 0:
        return 1.0
    return silent_frames / total_frames

def analyze_file(path, mode="speech", min_duration=0.3, max_duration=20.0):
    try:
        if is_corrupted(path):
            return (path, "Corrupted file")
        audio, sr = librosa.load(path, sr=None, mono=True)
        duration = len(audio) / sr
        if duration < min_duration or duration > max_duration:
            return (path, f"Duration out of range: {duration:.2f}s")
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-4:
            return (path, f"Too silent: RMS={rms:.6f}")
        sil_ratio = silent_ratio(audio)
        if sil_ratio > 0.6:
            return (path, f"Too much silence: {sil_ratio:.2%}")
        if mode == "speech" and not is_voice_present(audio, sr):
            return (path, "No speech detected")
        return None  # valid
    except Exception as e:
        return (path, f"Load error: {str(e)}")

def scan_dataset(folder, mode="speech", num_workers=8, quarantine_dir=None):
    audio_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    bad_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(analyze_file, path, mode): path for path in audio_files
        }
        for future in tqdm(as_completed(future_to_path), total=len(audio_files), desc=f"Scanning [{mode}]"):
            result = future.result()
            if result:
                path, reason = result
                bad_results.append((path, reason))
                if quarantine_dir:
                    rel_path = os.path.relpath(path, folder)
                    dst_path = os.path.join(quarantine_dir, rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.move(path, dst_path)
    return bad_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/data1/CCF2025/datasets_fullband/noisy_fullband/clean', help="Path to dataset folder")
    parser.add_argument("--output_txt", type=str, default="bad_files.txt", help="Output bad file list")
    parser.add_argument("--mode", choices=["speech", "noise"], default="speech", help="Cleaning mode")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of threads to use")
    parser.add_argument("--quarantine_dir", type=str, default="quarantine", help="Path to move bad files")
    args = parser.parse_args()

    bad_files = scan_dataset(
        folder=args.data_dir,
        mode=args.mode,
        num_workers=args.num_workers,
        quarantine_dir=args.quarantine_dir
    )

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for path, reason in bad_files:
            f.write(f"{path}|{reason}\n")

    print(f"\n✅ Done. Detected {len(bad_files)} bad files. See: {args.output_txt}")
