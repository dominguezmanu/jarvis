import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from scipy.spatial.distance import cdist

AUDIO_PATH = "audios"
COMMANDS = ["youtube", "terminal", "firefox", "servo"]

SAMPLE_RATE = 44100
DURATION = 2.5

N_MFCC = 13


def to_mono(audio):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)


def normalize(audio):
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio)) + 1e-9
    return audio / max_val


def trim_silence(audio, top_db=25):
    audio = normalize(audio)

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)

    if len(trimmed) < 1000:
        return audio

    return trimmed


def preprocess(audio):
    audio = to_mono(audio)
    audio = normalize(audio)
    audio = trim_silence(audio)
    audio = normalize(audio)
    return audio


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512
    )

    mfcc = mfcc.T

    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-9)

    return mfcc


def dtw_distance(mfcc1, mfcc2):
    distance_matrix = cdist(mfcc1, mfcc2, metric="cosine")

    D, wp = librosa.sequence.dtw(C=distance_matrix)

    total_distance = D[-1, -1]
    normalized_distance = total_distance / len(wp)

    return normalized_distance


def load_patterns():
    patterns = {}

    for cmd in COMMANDS:
        path = os.path.join(AUDIO_PATH, f"{cmd}.wav")

        if not os.path.exists(path):
            print(f"❌ No existe: {path}")
            continue

        audio, sr = sf.read(path)

        if sr != SAMPLE_RATE:
            audio = librosa.resample(to_mono(audio), orig_sr=sr, target_sr=SAMPLE_RATE)

        audio = preprocess(audio)
        mfcc = extract_mfcc(audio)

        patterns[cmd] = mfcc
        print(f"✔ Cargado: {cmd}")

    return patterns


def record_audio():
    print("\n🎤 Grabando... habla ahora")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("✔ Grabación lista")
    return audio.flatten()


def detect(audio, patterns):
    audio = preprocess(audio)
    mfcc_live = extract_mfcc(audio)

    scores = {}

    for cmd, pattern_mfcc in patterns.items():
        distance = dtw_distance(mfcc_live, pattern_mfcc)
        scores[cmd] = distance

    best = min(scores, key=scores.get)

    print("\n--- SCORES MFCC + DTW ---")
    print("Menor score = más parecido\n")

    for cmd, score in sorted(scores.items(), key=lambda x: x[1]):
        print(f"{cmd:10s}: {score:.4f}")

    print(f"\n🏆 Mejor coincidencia: {best} ({scores[best]:.4f})")


def main():
    print("📂 Cargando audios patrón...")
    patterns = load_patterns()

    if not patterns:
        print("No se cargó ningún audio.")
        return

    print("\nListo.")

    while True:
        input("\nPresiona ENTER para grabar...")
        audio = record_audio()
        detect(audio, patterns)


if __name__ == "__main__":
    main()