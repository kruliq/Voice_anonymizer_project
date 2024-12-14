import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, SpeechT5ForTextToSpeech, SpeechT5Processor
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

def synthesize_speech(text, tts_model, tts_processor, output_path, embeddings):
    inputs = tts_processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings=embeddings)
    
    # Save the synthesized speech to a WAV file
    sf.write(output_path, speech.squeeze().cpu().numpy(), 22050)

def main():
    # Load the pre-trained models and processors
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name)
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
    
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = embeddings_dataset[7306]["xvector"]
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    tts_model_name = "microsoft/speecht5_tts"
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_name)
    tts_processor = SpeechT5Processor.from_pretrained(tts_model_name)
    
    # Load the audio file
    input_file_path = "C:/Users/Krolik/Desktop/beznazwy.wav"
    output_file_path = "C:/Users/Krolik/Desktop/anon.wav"
    waveform, sample_rate = load_audio(input_file_path)
    
    # Transcribe the audio
    transcription = transcribe_audio(waveform, sample_rate, stt_model, stt_processor)
    print(f"Transcription: {transcription}")
    
    # Synthesize the anonymized speech
    synthesize_speech(transcription, tts_model, tts_processor, output_file_path, speaker_embeddings)
    print(f"Anonymized speech saved to {output_file_path}")

if __name__ == "__main__":
    main()