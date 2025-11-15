import torch
import torchaudio
import sounddevice as sd
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

wav, sampling_rate = torchaudio.load("assets/IU.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

while True:
    text = input("Text (type 'exit' to quit): ")
    if text == "exit":
        break
    cond_dict = make_cond_dict(text, speaker=speaker, language="ko")
    conditioning = model.prepare_conditioning(cond_dict)

    codes = model.generate(conditioning)

    wavs = model.autoencoder.decode(codes).cpu()
    sd.play(wavs[0].squeeze().numpy(), model.autoencoder.sampling_rate)
    sd.wait()