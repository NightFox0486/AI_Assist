import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np

# Initialize Whisper model (using larger model for better accuracy)
model = WhisperModel("medium", device="cpu", compute_type="int8")

# Audio recording parameters
duration = 5  # Recording duration in seconds
sample_rate = 16000  # Sample rate in Hz

# Function to record audio and transcribe
def record_and_transcribe():
    print("Recording... Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * sample_rate), 
                  samplerate=sample_rate, 
                  channels=1)
    sd.wait()
    
    # Check audio level
    audio_level = np.abs(audio).max()
    print(f"Audio level: {audio_level:.4f}")
    
    if audio_level < 0.001:
        print("⚠️  No audio detected. Please speak louder.")
        return
    
    # Convert audio to float32 format for Whisper
    audio = audio.flatten().astype(np.float32)
    
    # Transcribe audio using Whisper
    segments, info = model.transcribe(audio, 
                                    language="ko",
                                    beam_size=5,
                                    vad_filter=True,
                                    vad_parameters=dict(min_silence_duration_ms=500))
    
    # Print transcription
    transcription_found = False
    for segment in segments:
        print(f"Transcription: {segment.text}")
        transcription_found = True
    
    if not transcription_found:
        print("No speech detected in recording.")

if __name__ == "__main__":
    try:
        while True:
            input("Press Enter to start recording...")
            record_and_transcribe()
    except KeyboardInterrupt:
        print("\nProgram terminated.")