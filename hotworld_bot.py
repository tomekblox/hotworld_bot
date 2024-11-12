import wave
from transformers import *
import torch
import soundfile as sf
import os
import torchaudio
from pvrecorder import PvRecorder
import keyboard


class StoppedRecordingAudio(Exception):
    pass

#recording audio
for index, device in enumerate(PvRecorder.get_available_devices()):
    print(f"[{index}] {device}")
    
recorder = PvRecorder(device_index=-1, frame_length=512)
audio = []

try:
    recorder.start()
    print("start talking")

    while True:
        frame = recorder.read()
        audio.extend(frame)
        if keyboard.is_pressed('Space'):
            raise StoppedRecordingAudio
except StoppedRecordingAudio:
    print("stopped")
    recorder.stop()
    with wave.open("D:\\hotworld_bot\\audio_file.wav", 'w') as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(wave.struct.pack("h" * len(audio), *audio))
finally:
    recorder.delete()
    


#processing audio
device = "cuda:0" if torch.cuda.is_available() else "cpu"

wav2vec2_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"

# wav2vec2_model_name = "facebook/wav2vec2-base-960h"

wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name).to(device)

audio_url = "D:\\hotworld_bot\\audio_file.wav"

speech, sr = torchaudio.load(audio_url)
speech = speech.squeeze()

resampler = torchaudio.transforms.Resample(sr, 16000)
speech = resampler(speech)

input_values = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"].to(device)

logits = wav2vec2_model(input_values)["logits"]

predicted_ids = torch.argmax(logits, dim=-1)

transcription = wav2vec2_processor.decode(predicted_ids[0])
print(transcription.lower()) # processed textgith

input("Press Enter to continue...")