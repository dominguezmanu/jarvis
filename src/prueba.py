import sounddevice as sd
import soundfile as sf


print("Grabando test.wav")
audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1)
sd.wait()

sf.write("test.wav", audio, 16000)
print("Grabado test.wav")