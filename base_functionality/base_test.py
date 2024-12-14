import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, T5ForConditionalGeneration, T5Tokenizer
import torch
import soundfile as sf

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

def synthesize_speech(text, model, tokenizer, output_path):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Use a simple TTS model for demonstration (e.g., Tacotron2, WaveGlow)
    # Here we assume you have a TTS model that can generate waveform from text
    # For simplicity, we will save the generated text to a file
    with open(output_path, "w") as f:
        f.write(generated_text)

def main():
    # Load the pre-trained models and processors
    stt_model_name = "facebook/wav2vec2-base-960h"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name)
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
    
    tts_model_name = "t5-small"
    tts_model = T5ForConditionalGeneration.from_pretrained(tts_model_name)
    tts_tokenizer = T5Tokenizer.from_pretrained(tts_model_name)
    
    # Load the audio file
    input_file_path = "C:/Users/Krolik/Desktop/beznazwy.wav"
    output_file_path = "C:/Users/Krolik/Desktop/anon.wav"
    waveform, sample_rate = load_audio(input_file_path)
    
    # Transcribe the audio
    transcription = transcribe_audio(waveform, sample_rate, stt_model, stt_processor)
    print(f"Transcription: {transcription}")
    
    # Synthesize the anonymized speech
    synthesize_speech(transcription, tts_model, tts_tokenizer, output_file_path)
    print(f"Anonymized speech saved to {output_file_path}")

if __name__ == "__main__":
    main()