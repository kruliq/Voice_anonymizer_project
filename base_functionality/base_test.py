import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
from scipy.io.wavfile import write as wav_write
from datasets import load_dataset
import numpy as np

def load_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
    except RuntimeError as e:
        print(f"Error loading audio file: {e}")
        return None, None

def transcribe_audio(waveform, sample_rate, model, processor, device):
    if waveform is None or sample_rate is None:
        return "Error in loading audio file."

    # Resample the audio to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Process the audio
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def synthesize_speech(text, tts_model, tts_processor, output_path, embeddings, device):
    inputs = tts_processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings=embeddings.to(device))
    
    # Convert the speech tensor to numpy array and ensure it is in the correct format
    speech_np = speech.squeeze().cpu().numpy()
    
    # Save the synthesized speech to a WAV file with the correct sample rate
    wav_write(output_path, 22050, speech_np.astype('int16'))

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained models and processors
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
    
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = embeddings_dataset[7306]["xvector"]
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    tts_model_name = "microsoft/speecht5_tts"
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_name).to(device)
    tts_processor = SpeechT5Processor.from_pretrained(tts_model_name)
    
    # Load the audio file
    input_file_path = "C:/Users/Krolik/Desktop/test_pl.wav"
    output_file_path = "C:/Users/Krolik/Desktop/anon.wav"
    waveform, sample_rate = load_audio(input_file_path)
    
    # Get the original duration of the audio
    original_duration = waveform.shape[-1] / sample_rate
    
    # Transcribe the audio
    transcription = transcribe_audio(waveform, sample_rate, stt_model, stt_processor, device)
    print(f"Transcription: {transcription}")
    
    # Synthesize the anonymized speech
    synthesize_speech(transcription, tts_model, tts_processor, output_file_path, speaker_embeddings, device)
    print(f"Anonymized speech saved to {output_file_path}")

if __name__ == "__main__":
    main()