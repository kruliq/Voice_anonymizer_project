"""
Voice Anonymization System
-------------------------
This module implements a voice anonymization system using speech-to-text
and text-to-speech models. It preserves the content of speech while
modifying speaker characteristics.

Main components:
- Audio loading and preprocessing
- Speech transcription
- Text-to-speech synthesis
- Voice characteristic modification
"""

import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, VitsModel, AutoTokenizer
import torch
from scipy.io.wavfile import write as wav_write
from datasets import load_dataset
import getopt
import sys

import random
from audiostretchy.stretch import stretch_audio
import librosa
from scipy.io.wavfile import read  


import time
import os


def load_audio(file_path: str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file and return the waveform and sample rate.

    Args:
        file_path: Path to the audio file to load

    Returns:
        tuple: (waveform, sample_rate)
            - waveform: Audio waveform tensor of shape (channels, samples)
            - sample_rate: Sampling rate of the audio in Hz

    Example:
        >>> waveform, sample_rate = load_audio("speech.wav")
        >>> print(f"Audio duration: {waveform.shape[-1]/sample_rate:.2f}s")
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
    except RuntimeError as e:
        print(f"Error loading audio file: {e}")
        return None, None

def transcribe_audio(waveform: torch.Tensor, 
                    sample_rate: int, 
                    model: Wav2Vec2ForCTC, 
                    processor: Wav2Vec2Processor, 
                    device: torch.device) -> str:
    """
    Transcribe audio waveform to text using Wav2Vec2 model.

    Args:
        waveform: Audio waveform tensor of shape (channels, samples)
        sample_rate: Sampling rate of the audio in Hz
        model: Pre-trained Wav2Vec2 model for transcription
        processor: Wav2Vec2 processor for audio preprocessing
        device: Computation device (CPU/GPU)

    Returns:
        str: Transcribed text from audio

    Example:
        >>> text = transcribe_audio(waveform, 16000, model, processor, device)
        >>> print(f"Transcription: {text}")
    """
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

def get_speech_duration(waveform: torch.Tensor, 
                       sample_rate: int) -> float:
    """
    Calculate duration of speech in seconds.
    
    Args:
        waveform: Audio waveform tensor of shape (channels, samples)
        sample_rate: Sampling rate of the audio in Hz
    
    Returns:
        float: Duration of the speech in seconds

    Example:
        >>> duration = get_speech_duration(waveform, sample_rate)
        >>> print(f"Duration: {duration:.2f}s")
    """
    return waveform.shape[-1] / sample_rate

def normalize_length(waveform: torch.Tensor, 
                    target_length: int, 
                    sample_rate: int) -> torch.Tensor:
    sr1 = sample_rate
    """
    Adjust waveform length without changing pitch using resampling.
    
    Args:
        waveform: Audio waveform tensor to adjust
        target_length: Desired length in samples
        sample_rate: Sampling rate of the audio in Hz
    
    Returns:
        torch.Tensor: Length-normalized waveform

    Example:
        >>> normalized_waveform = normalize_length(waveform, target_length, sample_rate)
        >>> print(f"New length: {normalized_waveform.shape[-1]}")
    """
    current_length = waveform.shape[-1]
    if current_length == target_length:
        return waveform
    return waveform  
    # Calculate new sampling rate for time stretching
    """
    stretch_factor = target_length / current_length
    new_sample_rate = int(sample_rate * stretch_factor)
    
    # Apply resampling for time stretching
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=new_sample_rate
    )
    return resampler(waveform)
    """

def process_audio(audio_input, stretch_ratio=0.8, pitch_steps=0, sample_rate=16000):
    """
    Process audio with stretching and pitch shifting.
    
    Args:
        audio_input: Input audio array
        stretch_ratio: Time stretching ratio (default: 0.8)
        pitch_steps: Number of steps for pitch shifting (default: 0)
        sample_rate: Audio sample rate (default: 16000)
    
    Returns:
        processed_audio: Processed audio array
    """
    try:
        # Convert audio to int16
        audio_to_stretch = (audio_input * 32767).numpy().astype('int16')
        
        # Create temporary file for stretching
        temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads/temp_process.wav')
        wav_write(temp_file, sample_rate, audio_to_stretch)
        os.chmod(temp_file, 0o644)
        print("stretch_ratio: ", stretch_ratio)
        # Apply time stretching
        stretch_audio(temp_file, temp_file, ratio=stretch_ratio)
        
        # Load stretched audio for pitch shifting
        audio_stretched, sr = librosa.load(temp_file, sr=sample_rate)
        
        # Apply pitch shifting
        processed_audio = librosa.effects.pitch_shift(
            audio_stretched, 
            sr=sr, 
            n_steps=pitch_steps
        )
        
        return processed_audio
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_file):
            os.remove(temp_file)

def synthesize_speech(text: str, 
                     model: VitsModel, 
                     tokenizer: AutoTokenizer, 
                     output_path: str, 
                     device: torch.device, 
                     target_duration: float = None) -> None:
    """
    Synthesize speech from text with optional duration matching.
    
    Args:
        text: Input text to synthesize
        model: Pre-trained TTS model
        tokenizer: Text tokenizer for the model
        output_path: Path to save the output audio file
        device: Computation device (CPU/GPU)
        target_duration: Optional target duration in seconds

    Example:
        >>> synthesize_speech("Hello world", model, tokenizer, "output.wav", device, 5.0)
    """
    # Prepare input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate speech
    with torch.no_grad():
        output = model(**inputs).waveform
        audio = output.squeeze().cpu()
    audio_length_seconds = audio.shape[-1] / model.config.sampling_rate
    stretch_factor = target_duration / audio_length_seconds
    print("target_duration: ", target_duration, "audio_length_seconds: ", audio_length_seconds, "stretch_factor: ", stretch_factor)
    # Process audio with stretching and pitch shifting
    processed_audio = process_audio(
        audio,
        stretch_ratio=stretch_factor,
        pitch_steps=0,
        sample_rate=16000
    )
    
    if processed_audio is not None:
        # Save the synthesized speech
        wav_write(
            output_path,
            rate=model.config.sampling_rate,
            data=processed_audio
        )
    
def main():
    """
    Main function to run the voice anonymization system.
    
    Steps:
        1. Load audio file
        2. Extract speaker embeddings
        3. Transcribe audio to text
        4. Synthesize speech with modified voice characteristics
        5. Save the output audio file

    Example:
        >>> main()
    """
    start = time.time()
    
    # Remove the first argument from the list of command line arguments
    argument_list = sys.argv[1:]

    # Options
    options = "i:o:"

    # Long options
    long_options = ["input=", "output="]

    input_file = "base_functionality/web/uploads/test_pl.wav"
    output_file = "base_functionality/web/uploads/test_pl_anon.wav"

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argument_list, options, long_options)
        
        # Checking each argument
        for current_argument, current_value in arguments:
            if current_argument in ("-i", "--input"):
                input_file = current_value
            elif current_argument in ("-o", "--output"):
                output_file = current_value

        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")

    except getopt.error as err:
        # Output error, and return with an error code
        print(str(err))
        return

    load_start = time.time()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
    
    # Load the embeddings dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7073]["xvector"]).unsqueeze(0).to(device)

    # Load the TTS model and tokenizer
    tts_model_name = "facebook/mms-tts-pol"
    tts_model = VitsModel.from_pretrained(tts_model_name).to(device)
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
    
    # Load the audio file
    waveform, sample_rate = load_audio(input_file)
       
    load_end = time.time()
    print('Models loading time: ', load_end-load_start, 's')

   
    
    # Get the original duration of the audio
    original_duration = waveform.shape[-1] / sample_rate
    
    # Transcribe the audio
    transcription = transcribe_audio(waveform, sample_rate, stt_model, stt_processor, device)
    print(f"Transcription: {transcription}")
    
    # Get input speech duration
    input_duration = get_speech_duration(waveform, sample_rate)
    print(f"Input duration: {input_duration:.2f}s")
    
    # Synthesize with matching duration
    synthesize_speech(
        transcription,
        tts_model, 
        tts_tokenizer,
        output_file,
        device,
        target_duration=original_duration
    )
    print(f"Anonymized speech saved to {output_file}")
    
    end = time.time()
    time_diff = end - start
    print(f"Working time: {time_diff:.6f}s")
    RTF = time_diff/input_duration
    print(f"RealTimeFactor: {RTF:.6f}")
    print(input_duration, time_diff, RTF, sep='\t')

if __name__ == "__main__":
    main()
