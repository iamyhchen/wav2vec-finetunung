from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

model = Wav2Vec2ForCTC.from_pretrained("model_id")
processor = Wav2Vec2Processor.from_pretrained("model_id")

audio_path = "path/to/wav.wav"
audio_input, sr = librosa.load(audio_path, sr=16000)

inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(inputs.input_values).logits
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]

print("辨識結果：", transcription)