import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import serial
import subprocess
from scipy.spatial.distance import cdist

# ================= CONFIG =================

AUDIO_PATH = "audios"
COMMANDS = ["youtube", "visual", "firefox", "servo"]

#para el puerto serial
SERIAL_PORT = "/dev/ttyUSB0"
BAUD = 115200

#para e audio
SAMPLE_RATE = 44100
DURATION = 2.5
N_MFCC = 13

#Correlacion minima que debe tener para aceptar un comando
MAX_ACCEPTED_DISTANCE = 0.45


# convierte audio a mono, porque se grabaron en mono y por si se graba en stereo
def convertir_a_mono(audio):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)

#volumen de -1 a 1
def normalizar(audio):
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio)) + 1e-9
    return audio / max_val


#Elimina los silencios al inicio y final del audio para mejor comparacin
def recortar_silencio(audio, top_db=25):
    audio = normalizar(audio)
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)

    if len(trimmed) < 1000:
        return audio

    return trimmed

#esto hace que el audio pase por todo el proceso de preparacion
def preprocesar(audio):
    audio = convertir_a_mono(audio)
    audio = normalizar(audio)
    audio = recortar_silencio(audio)
    audio = normalizar(audio)
    return audio

#extrae los mfcc del audio para una mejor comparacion
def extraer_mfcc(audio):
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

#Compara los audios  usando DTW, saber como lo hace la verda, esto fue sacado de la docuemntacion xd
def distancia_dtw(mfcc1, mfcc2):
    distance_matrix = cdist(mfcc1, mfcc2, metric="cosine")
    D, wp = librosa.sequence.dtw(C=distance_matrix)

    total_distance = D[-1, -1]
    normalized_distance = total_distance / len(wp)

    return normalized_distance

#aqui se cargan los patrones de audio y se hacen pasar por procesamiento, por uniformidad 
def cargar_patrones():
    patterns = {}

    for cmd in COMMANDS:
        path = os.path.join(AUDIO_PATH, f"{cmd}.wav")

        if not os.path.exists(path):
            print(f" No existe: {path}")
            continue

        audio, sr = sf.read(path)

        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                convertir_a_mono(audio),
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

        audio = preprocesar(audio)
        mfcc = extraer_mfcc(audio)

        patterns[cmd] = mfcc
        print(f"✔ Cargado: {cmd}")

    return patterns


#Graba el audio
def grabar_audio():
    print("\n Grabando... habla ahora")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    print("✔ Grabación lista")

    return audio.flatten()


#Limpia el audio grabado, caclula los mfcc y compara con los patrones
# guarda el score mas bajo como el mejor
def detectar(audio, patterns):
    audio = preprocesar(audio)
    mfcc_live = extraer_mfcc(audio)

    scores = {}

    for cmd, pattern_mfcc in patterns.items():
        distance = distancia_dtw(mfcc_live, pattern_mfcc)
        scores[cmd] = distance

    best = min(scores, key=scores.get)
    best_score = scores[best]

    print("\n--- score de correlacion---")


    for cmd, score in sorted(scores.items(), key=lambda x: x[1]):
        print(f"{cmd:10s}: {score:.4f}")

    print(f"\nMejor coincidencia: {best} ({best_score:.4f})")

    if best_score <= MAX_ACCEPTED_DISTANCE:
        return best

    return None


# acciones en consola, son puros comando, excepto el ultimo
#que manda "servo" por el serial pal arduino
def ejecutar_accion(command, ser):
    if command == "firefox":
        print("Abriendo Firefox")
        subprocess.Popen(["firefox"])

    elif command == "visual":
        print("Abriendo VS Code")
        subprocess.Popen(["code"])

    elif command == "youtube":
        print("Abriendo YouTube")
        subprocess.Popen(["xdg-open", "https://www.youtube.com/watch?v=glaFn_mdqdQ"])

    elif command == "servo":
        print("Enviando comando al arduino...")
        ser.write(b"SERVO\n")


# funcion main

def principal():
    print("Conectando alArduino")
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)

    print("Cargando audios patrón")
    patterns = cargar_patrones()

    if not patterns:
        print("No se cargó ningún audio.")
        return

    print("\nSistema listo.")
    print("Presiona el botón del para grabar")

    while True:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()

            if line == "RECORD":
                print("\nBotón presionado.")

                audio = grabar_audio()
                result = detectar(audio, patterns)

                if result:
                    print(f"omando detectado: {result}")
                    ejecutar_accion(result, ser)
                else:
                    print("NO reconocido, saber que dijo compadre")

        time.sleep(0.05)


if __name__ == "__main__":
    principal()