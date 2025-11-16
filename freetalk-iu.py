import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
sys.path.append('RealTime_zeroshot_TTS_ko')

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import mecab_wrapper
from custom_tts import Custom_TTS
import scipy.io.wavfile as wavfile

# Initialize Whisper (balanced speed/accuracy)
print("Loading Whisper model...")
whisper_model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    num_workers=4
)

# Initialize OpenVoice TTS with IU voice
print("Loading OpenVoice TTS...")
os.chdir('RealTime_zeroshot_TTS_ko')
tts = Custom_TTS()
tts.set_model(language='KR')
tts.get_reference_speaker('../assets/IU.mp3', vad=True)
os.chdir('..')
print("Ready!")

# Audio parameters
sample_rate = 16000
recording_duration = 3

def record_and_transcribe():
    """Record audio and transcribe with improved accuracy"""
    print(f"\nRecording for {recording_duration} seconds... Speak now!")
    
    audio = sd.rec(int(recording_duration * sample_rate), 
                  samplerate=sample_rate, 
                  channels=1,
                  dtype='float32')
    sd.wait()
    
    audio_level = np.abs(audio).max()
    print(f"Audio level: {audio_level:.4f}")
    
    if audio_level < 0.001:
        print("⚠️  No audio detected.")
        return None
    
    audio = audio.flatten()
    
    print("Transcribing...")
    segments, info = whisper_model.transcribe(
        audio,
        language="ko",
        beam_size=3,
        best_of=3,
        temperature=0.0,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            threshold=0.5
        ),
        condition_on_previous_text=False
    )
    
    text = ""
    for segment in segments:
        text += segment.text
    
    return text.strip()

def text_to_speech(text):
    """Convert text to speech using OpenVoice TTS"""
    if not text:
        return
    
    print(f"You said: {text}")
    print("Generating speech...")
    
    original_dir = os.getcwd()
    os.chdir('RealTime_zeroshot_TTS_ko')
    
    current_count = tts.result_cnt
    tts.make_speech(text, speed=1.1)
    
    output_file = f'output/result_{current_count}.wav'
    sr, audio_data = wavfile.read(output_file)
    os.chdir(original_dir)
    
    print("Playing audio...")
    audio_float = audio_data.astype(np.float32) / 32768.0
    sd.play(audio_float, sr)
    sd.wait()
    
    os.remove(f'RealTime_zeroshot_TTS_ko/{output_file}')

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
