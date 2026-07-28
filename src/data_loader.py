import soundfile as sf

def load_audio(filepath):
    audio, sr = sf.read(filepath)
    return audio, sr