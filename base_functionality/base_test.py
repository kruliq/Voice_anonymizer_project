import os
import torchaudio
import torch
import numpy as np
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5Processor
)
from scipy.io.wavfile import write as wav_write
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

def load_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
    except RuntimeError as e:
        print(f"Error loading audio file: {e}")
        return None, None

def extract_speaker_embeddings(waveform, sample_rate, device):
    try:
        bundle = WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
        
        if sample_rate != bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=bundle.sample_rate
            )
            waveform = resampler(waveform)
        
        with torch.no_grad():
            features, _ = model.extract_features(waveform.to(device))
            # Reshape embeddings to match SpeechT5 requirements
            embeddings = features[-1].mean(dim=1)  # [batch_size, hidden_size]
            
            # Create fixed projection matrix
            projection_matrix = torch.nn.Parameter(
                torch.randn(1280, 768, device=device) / np.sqrt(1280)
            )
            
            # Project from 1x1280 to 1x768
            embeddings = torch.matmul(embeddings, projection_matrix)
            
            print(f"Embedding shape: {embeddings.shape}")  # Debug info
            return embeddings

    except Exception as e:
        print(f"Error extracting speaker embeddings: {e}")
        return None

def transcribe_audio(waveform, sample_rate, model, processor, device):
    if waveform is None or sample_rate is None:
        return "Error in loading audio file."

    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Process audio
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def synthesize_speech(text, model, processor, output_path, speaker_embeddings, device):
    inputs = processor(text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings
        )
    
    audio_data = output.squeeze().cpu().numpy()
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    wav_write(output_path, model.config.sampling_rate, audio_data)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
    
    tts_model_name = "microsoft/speecht5_tts"
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_name).to(device)
    tts_processor = SpeechT5Processor.from_pretrained(tts_model_name)
    
    # Process audio
    input_file_path = "C:/Users/Krolik/Desktop/test_pl.wav"
    output_file_path = "C:/Users/Krolik/Desktop/anon.wav"
    waveform, sample_rate = load_audio(input_file_path)
    
    if waveform is None:
        print("Failed to load audio file")
        return
    
    speaker_embeddings = extract_speaker_embeddings(waveform, sample_rate, device)
    if speaker_embeddings is None:
        print("Failed to extract speaker embeddings")
        return
    
    transcription = transcribe_audio(waveform, sample_rate, stt_model, stt_processor, device)
    print(f"Transcription: {transcription}")
    
    synthesize_speech(transcription, tts_model, tts_processor, output_file_path, speaker_embeddings, device)
    print(f"Anonymized speech saved to {output_file_path}")

if __name__ == "__main__":
    main()