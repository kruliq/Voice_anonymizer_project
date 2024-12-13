import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def load_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
    except RuntimeError as e:
        print(f"Error loading audio file: {e}")
        return None, None

def transcribe_audio(waveform, sample_rate, model, processor):
    if waveform is None or sample_rate is None:
        return "Error in loading audio file."

    # Resample the audio to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Process the audio
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def main():
    # Load the pre-trained model and processor
    model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    # Load the audio file
    file_path = "C:/Users/Krolik/Desktop/beznazwy.wav"
    waveform, sample_rate = load_audio(file_path)
    
    # Transcribe the audio
    transcription = transcribe_audio(waveform, sample_rate, model, processor)
    
    # Print the transcription
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()