import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import torch
import soundfile as sf
from datasets import load_dataset

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

def synthesize_speech(text, synthesiser, speaker_embedding, output_path):
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained models and processors
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
    
    # Load the embeddings dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    # Initialize the text-to-speech pipeline
    synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts", device=device.index if device.type == "cuda" else -1)
    
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
    synthesize_speech(transcription, synthesiser, speaker_embedding, output_file_path)
    print(f"Anonymized speech saved to {output_file_path}")

if __name__ == "__main__":
    main()