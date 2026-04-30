import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal

# ================= CONFIGURACION =================

AUDIO_PATH = "audios"
COMMANDS = ["youtube", "visual", "firefox", "servo"]

SAMPLE_RATE = 44100
DURATION = 2.5


# --------- PREPROCESAMIENTO ---------

def convertir_a_mono(audio):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)


def normalizar(audio):
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio)) + 1e-9
    return audio / max_val


def recortar_silencio(audio, threshold=0.02):
    audio = normalizar(audio)
    mask = np.abs(audio) > threshold

    if not np.any(mask):
        return audio

    start = np.argmax(mask)
    end = len(audio) - np.argmax(mask[::-1])
    return audio[start:end]


def reasignar_si_necesario(audio, sr):
    if sr == SAMPLE_RATE:
        return audio

    new_len = int(len(audio) * SAMPLE_RATE / sr)
    return signal.resample(audio, new_len)


def preprocesar(audio, sr):
    audio = convertir_a_mono(audio)
    audio = reasignar_si_necesario(audio, sr)
    audio = recortar_silencio(audio)
    audio = normalizar(audio)
    return audio


# --------- CORRELACIÓN SIMPLE ---------

def correlacion_simple(a, b):
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]

    if len(a) < 100:
        return 0

    corr = np.correlate(a, b, mode="valid")[0]

    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9

    return corr / norm


# --------- CARGA DE AUDIOS ---------

def cargar_patrones():
    patterns = {}

    for cmd in COMMANDS:
        path = os.path.join(AUDIO_PATH, f"{cmd}.wav")

        if not os.path.exists(path):
            print(f" No existe: {path}")
            continue

        audio, sr = sf.read(path)
        audio = preprocesar(audio, sr)

        patterns[cmd] = audio
        print(f"Cargado: {cmd}")

    return patterns


# --------- GRABACIÓN ---------

def grabar_audio():
    print("\nGrabando... habla ahora")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    print("Grabación lista")

    return audio.flatten()


# --------- DETECCIÓN ---------

def detectar(audio, patterns):
    audio = preprocesar(audio, SAMPLE_RATE)

    scores = {}

    for cmd, pattern in patterns.items():
        score = correlacion_simple(audio, pattern)
        scores[cmd] = score

    best = max(scores, key=scores.get)

    print("\n--- SCORES CORRELACIÓN SIMPLE ---")
    print("Mayor score = más parecido\n")

    for cmd, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{cmd:10s}: {score:.4f}")

    print(f"\nMejor coincidencia: {best} ({scores[best]:.4f})")


# --------- MAIN ---------

def principal():
    print("Cargando audios patrón")
    patterns = cargar_patrones()

    if not patterns:
        print("No se cargó ningún audio")
        return

    print("\nListo.")

    while True:
        input("\nPresiona ENTER para grabar")
        audio = grabar_audio()
        detectar(audio, patterns)


if __name__ == "__main__":
    principal()