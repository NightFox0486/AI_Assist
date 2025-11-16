import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

import sounddevice as sd
import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
import queue
import threading
import time

# Initialize models
print("Loading Whisper model...")
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

print("Loading TTS model...")
tts_model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

print("Loading speaker sample...")
wav, sampling_rate = torchaudio.load("assets/IU.mp3")
speaker = tts_model.make_speaker_embedding(wav, sampling_rate)

# Audio parameters
sample_rate = 16000
recording_duration = 5

# Create queues
text_queue = queue.Queue()

def record_and_transcribe():
    """Record audio and transcribe"""
    print(f"\nRecording for {recording_duration} seconds... Speak now!")
    
    audio = sd.rec(int(recording_duration * sample_rate), 
                  samplerate=sample_rate, 
                  channels=1)
    sd.wait()
    
    audio_level = np.abs(audio).max()
    print(f"Audio level: {audio_level:.4f}")
    
    if audio_level < 0.001:
        print("⚠️  No audio detected.")
        return None
    
    audio = audio.flatten().astype(np.float32)
    
    print("Transcribing...")
    segments, info = whisper_model.transcribe(audio, 
                                             language="ko",
                                             beam_size=5,
                                             vad_filter=True)
    
    text = ""
    for segment in segments:
        text += segment.text
    
    return text.strip()

def text_to_speech(text):
    """Convert text to speech and play"""
    if not text:
        return
    
    print(f"You said: {text}")
    print("Generating speech...")
    
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="ko")
    conditioning = tts_model.prepare_conditioning(cond_dict)
    codes = tts_model.generate(conditioning)
    wavs = tts_model.autoencoder.decode(codes).cpu()
    
    print("Playing audio...")
    audio_data = wavs[0].squeeze().numpy()
    sd.play(audio_data, tts_model.autoencoder.sampling_rate)
    sd.wait()

print("\nReady! Press Enter to start recording, or type 'exit' to quit.")

try:
    while True:
        user_input = input("\nPress Enter to record: ")
        if user_input.lower() == 'exit':
            break
        
        text = record_and_transcribe()
        if text:
            text_to_speech(text)
        
except KeyboardInterrupt:
    print("\nProgram terminated.")